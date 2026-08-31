"""Default Bedrock Mantle routing for the OpenAI GPT-5.6 family, and its price.

``aws_bedrock_mantle_preferred_models`` defaults to the GPT-5.6 family, so a
deployment configuring nothing serves Sol, Terra and Luna from Bedrock Mantle --
the only endpoint carrying their web search and code interpreter tools. Three
consequences are asserted here because none of them is visible in a request:
the routing itself, the startup refusal that stops a guardrail being dropped
silently, and the 10% per-token price rise that follows Mantle having no
cross-Region inference profile to discount.

The model IDs below were verified dual-homed live in ``us-east-1`` on
2026-08-26 (``bedrock:ListFoundationModels`` intersected with the Mantle
``/v1/models`` catalogue).

Ref: stdapi/config.py:DEFAULT_MANTLE_PREFERRED_MODELS
     stdapi/models/__init__.py:is_mantle_preferred
     stdapi/models/pricing_overrides.py:DEFAULT_MODEL_GLOBAL_PRICES
     https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from re import findall
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from stdapi import models
from stdapi.batches import _batch_model_id
from stdapi.config import DEFAULT_MANTLE_PREFERRED_MODELS, SETTINGS, _Settings
from stdapi.models import (
    MANTLE_MODELS,
    MANTLE_SERVICE,
    RUNTIME_SERVICE,
    SAGEMAKER_SERVICE,
    ModelDetails,
    _merge_mantle_models,
    is_mantle_preferred,
)
from stdapi.models.chat import get_chat_model, serves_via_mantle
from stdapi.models.pricing_overrides import (
    DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES,
    DEFAULT_MODEL_GLOBAL_PRICES,
    DEFAULT_MODEL_LONG_CONTEXT_PRICES,
    DEFAULT_MODEL_PRICES,
)
from stdapi.monitoring import TENANT, Tenant, TenantAwsCredential
from stdapi.pricing import (
    Dimension,
    Service,
    _apply_default_prices,
    _state,
    resolve_model_key,
    resolve_price,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.pricing import Price, PriceKey

pytestmark = pytest.mark.local

#: GPT-5.6 model IDs verified served by both bedrock-runtime and Bedrock Mantle.
_DUAL_HOMED = ("openai.gpt-5.6-sol", "openai.gpt-5.6-terra", "openai.gpt-5.6-luna")

#: A region the GPT-5.6 model-card rates are published for.
_REGION: RegionName = "us-east-1"

#: A region only the bedrock-runtime home of a displaced dual-homed model serves.
_RUNTIME_REGION: RegionName = "us-west-2"

#: The cross-Region inference profile that runtime home is invoked through.
_RUNTIME_PROFILE = "us.openai.gpt-5.6-sol"

#: The one page quoting the per-million figures in prose; every other links to it.
_PRICE_REFERENCE = Path(__file__).parents[1] / "docs" / "operations_configuration.md"

#: The sentence in that page carrying the figures.
_PRICE_SENTENCE_MARKER = "exactly 10% more per token"

#: A syntactically valid per-end-user role ARN, in no real account.
_ROLE_ARN = "arn:aws:iam::123456789012:role/stdapi-ai-end-user"


def _snapshot_mantle_registries() -> tuple[
    dict[str, ModelDetails], dict[str, ModelDetails]
]:
    """Copy the process-global registries ``_merge_mantle_models`` rewrites.

    Returns:
        The Mantle registry, and the bedrock-runtime entries it displaced.
    """
    return dict(MANTLE_MODELS), dict(models._DISPLACED_RUNTIME_MODELS)  # noqa: SLF001


def _restore_mantle_registries(
    snapshot: tuple[dict[str, ModelDetails], dict[str, ModelDetails]],
) -> None:
    """Put both registries back exactly as they were snapshotted.

    Re-running the merge cannot rebuild ``_DISPLACED_RUNTIME_MODELS``: an entry
    is recorded only while the bedrock-runtime one is still the entry being
    overwritten, which an already-merged catalogue no longer carries. So the
    snapshot is written back rather than re-derived -- otherwise every later
    test on the worker reads the session catalogue with its runtime homes gone,
    and the tenant pin, the twin index and the batch gate answer as if the
    models were Mantle-only.

    Args:
        snapshot: What :func:`_snapshot_mantle_registries` returned.
    """
    mantle, displaced = snapshot
    MANTLE_MODELS.clear()
    MANTLE_MODELS.update(mantle)
    models._DISPLACED_RUNTIME_MODELS.clear()  # noqa: SLF001
    models._DISPLACED_RUNTIME_MODELS.update(displaced)  # noqa: SLF001


class TestDefaultRouting:
    """The family reaches Mantle with no configuration, and comes back with one."""

    def test_default_is_the_gpt_5_6_family(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unconfigured deployment prefers Mantle for the GPT-5.6 family.

        The test environment sets the variable, so it is removed first: this
        asserts the shipped default, not the suite's own configuration.
        """
        monkeypatch.delenv("aws_bedrock_mantle_preferred_models", raising=False)
        assert _Settings().aws_bedrock_mantle_preferred_models == [
            *DEFAULT_MANTLE_PREFERRED_MODELS
        ]

    @pytest.mark.parametrize("model_id", _DUAL_HOMED)
    def test_every_dual_homed_member_is_preferred(
        self, model_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each verified dual-homed GPT-5.6 ID resolves to Mantle under the default.

        The default is a prefix, so a member AWS adds to the family later is
        covered without a release.
        """
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_mantle_preferred_models",
            [*DEFAULT_MANTLE_PREFERRED_MODELS],
        )
        assert is_mantle_preferred(model_id)

    def test_other_dual_homed_models_stay_on_bedrock_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default moves the GPT-5.6 family alone, not every dual-homed model.

        Thirty-four other model IDs were live in both catalogues on the same
        day; a default that captured them would change the price and the
        guardrail posture of most of the frontier catalogue.
        """
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_mantle_preferred_models",
            [*DEFAULT_MANTLE_PREFERRED_MODELS],
        )
        assert not is_mantle_preferred("anthropic.claude-opus-5")
        assert not is_mantle_preferred("openai.gpt-oss-safeguard-120b")
        assert not is_mantle_preferred("openai.gpt-5.5")

    def test_an_empty_value_restores_runtime_twin_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting the variable empty puts every dual-homed model back on bedrock-runtime.

        This is the documented way out of the price and guardrail change, and it
        has to be asserted through the environment: ``env_ignore_empty`` reads an
        empty variable as unset, so the default would silently reapply. The
        init-keyword path never reaches that source and would pass either way.

        Ref: stdapi/config.py:_keep_explicit_empty
        """
        monkeypatch.delenv("aws_bedrock_mantle_preferred_models", raising=False)
        monkeypatch.setenv("AWS_BEDROCK_MANTLE_PREFERRED_MODELS", "")
        assert _Settings().aws_bedrock_mantle_preferred_models == []

        monkeypatch.delenv("AWS_BEDROCK_MANTLE_PREFERRED_MODELS")
        assert _Settings().aws_bedrock_mantle_preferred_models == list(
            DEFAULT_MANTLE_PREFERRED_MODELS
        )

        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_preferred_models", [])
        assert not is_mantle_preferred("openai.gpt-5.6-sol")

    @pytest.mark.xdist_group("model_cache")
    def test_the_merge_serves_a_dual_homed_model_from_the_preferred_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The catalogue publishes the Mantle entry for a preferred dual-homed model.

        ``_merge_mantle_models`` is where the preference becomes the served
        model: bedrock-runtime otherwise keeps a dual-homed ID. It also
        clears and rewrites the process-global ``MANTLE_MODELS`` registry
        every other module reads, so this test saves and restores it -- the
        same module singleton tests/test_model_cache.py isolates, hence the
        shared ``xdist_group``.

        Ref: stdapi/models/__init__.py:_merge_mantle_models
        """

        def details(service: str | None) -> ModelDetails:
            model = ModelDetails(
                id="openai.gpt-5.6-sol",
                name="GPT-5.6 Sol",
                provider="OpenAI",
                input_modalities=["TEXT"],
                output_modalities=["TEXT"],
                regions=[_REGION],
            )
            if service is not None:
                model.service = service
            return model

        def catalogues() -> tuple[dict[str, ModelDetails], dict[str, ModelDetails]]:
            return (
                {"openai.gpt-5.6-sol": details(None)},
                {"openai.gpt-5.6-sol": details(MANTLE_SERVICE)},
            )

        saved_registries = _snapshot_mantle_registries()
        try:
            monkeypatch.setattr(
                SETTINGS,
                "aws_bedrock_mantle_preferred_models",
                [*DEFAULT_MANTLE_PREFERRED_MODELS],
            )
            runtime, mantle = catalogues()
            _merge_mantle_models(runtime, mantle)
            assert runtime["openai.gpt-5.6-sol"].service == MANTLE_SERVICE

            monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_preferred_models", [])
            runtime, mantle = catalogues()
            _merge_mantle_models(runtime, mantle)
            assert runtime["openai.gpt-5.6-sol"].service != MANTLE_SERVICE
        finally:
            _restore_mantle_registries(saved_registries)

    def test_the_merge_test_does_not_leak_mantle_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The test above must not permanently corrupt the shared registry.

        ``_merge_mantle_models`` clears and rewrites the process-global
        ``MANTLE_MODELS`` unconditionally, so driving it from a test without
        saving and restoring the registry first corrupts it for every later
        reader on the same worker (stdapi/models/chat/__init__.py,
        stdapi/routes/anthropic_messages.py, tests/test_bedrock_mantle_live.py).
        A sentinel proves the registry comes back exactly as found, regardless
        of what the test above did to it in between. Seeding the sentinel is
        itself a rewrite of the shared registry, so this test saves and
        restores it exactly as the ones it checks do.
        """
        sentinel = {
            "sentinel-model": ModelDetails(
                id="sentinel-model",
                name="Sentinel",
                provider="Sentinel",
                input_modalities=["TEXT"],
                output_modalities=["TEXT"],
                regions=[_REGION],
            )
        }
        saved_registries = _snapshot_mantle_registries()
        MANTLE_MODELS.clear()
        MANTLE_MODELS.update(sentinel)
        try:
            self.test_the_merge_serves_a_dual_homed_model_from_the_preferred_endpoint(
                monkeypatch
            )
            assert sentinel == MANTLE_MODELS
        finally:
            _restore_mantle_registries(saved_registries)

    @pytest.mark.xdist_group("model_cache")
    def test_a_displaced_model_keeps_its_bedrock_runtime_display_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A preferred dual-homed model is published under its runtime display name.

        ``bedrock:ListFoundationModels`` is the only one of the two catalogues
        carrying a human-readable ``modelName``: the Mantle ``/v1/models``
        entries have none, so ``_get_mantle_models_from_region`` names them
        after the identifier. Serving the family from Mantle therefore replaces
        "GPT-5.6 Sol" with "gpt-5.6-sol" everywhere the catalogue is published,
        the public Models page included, unless the merge carries the displaced
        name over. A Mantle-only model has no name to inherit and keeps its own.

        Ref: stdapi/models/__init__.py:_merge_mantle_models
             stdapi/models/__init__.py:_get_mantle_models_from_region
        """

        def details(model_id: str, name: str, service: str | None) -> ModelDetails:
            model = ModelDetails(
                id=model_id,
                name=name,
                provider="OpenAI",
                input_modalities=["TEXT"],
                output_modalities=["TEXT"],
                regions=[_REGION],
            )
            if service is not None:
                model.service = service
            return model

        saved_registries = _snapshot_mantle_registries()
        try:
            monkeypatch.setattr(
                SETTINGS,
                "aws_bedrock_mantle_preferred_models",
                [*DEFAULT_MANTLE_PREFERRED_MODELS],
            )
            runtime = {
                "openai.gpt-5.6-sol": details("openai.gpt-5.6-sol", "GPT-5.6 Sol", None)
            }
            mantle = {
                "openai.gpt-5.6-sol": details(
                    "openai.gpt-5.6-sol", "gpt-5.6-sol", MANTLE_SERVICE
                ),
                "openai.gpt-5.6-nimbus": details(
                    "openai.gpt-5.6-nimbus", "gpt-5.6-nimbus", MANTLE_SERVICE
                ),
            }
            _merge_mantle_models(runtime, mantle)

            served = runtime["openai.gpt-5.6-sol"]
            assert served.service == MANTLE_SERVICE
            assert served.name == "GPT-5.6 Sol"
            assert MANTLE_MODELS["openai.gpt-5.6-sol"].name == "GPT-5.6 Sol"
            assert runtime["openai.gpt-5.6-nimbus"].name == "gpt-5.6-nimbus"
        finally:
            _restore_mantle_registries(saved_registries)


def _details(model_id: str, name: str, service: str) -> ModelDetails:
    """Build one catalogue entry on *service*.

    Args:
        model_id: The model identifier.
        name: The display name the catalogue carries.
        service: The hosting service label.

    Returns:
        The entry.
    """
    model = ModelDetails(
        id=model_id,
        name=name,
        provider="OpenAI",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=[_REGION],
    )
    model.service = service
    return model


@pytest.fixture
def displaced_runtime_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Build the catalogue a preferred, dual-homed model leaves behind.

    Both catalogues name a dual-homed GPT-5.6 model *identically*, so the merge
    overwrites the bedrock-runtime entry in place rather than adding a second
    one under another identifier. Everything derived from the catalogue
    afterwards -- the twin index, the tenant service pin, the batch
    advertisement -- sees only what the merge left behind.

    Yields:
        The dual-homed model identifier.
    """
    monkeypatch.setattr(
        SETTINGS,
        "aws_bedrock_mantle_preferred_models",
        [*DEFAULT_MANTLE_PREFERRED_MODELS],
    )
    model_id = _DUAL_HOMED[0]
    saved_models = dict(models._MODELS)  # noqa: SLF001
    saved_registries = _snapshot_mantle_registries()
    runtime_entry = _details(model_id, "GPT-5.6 Sol", RUNTIME_SERVICE)
    # The runtime home answers in a region Mantle does not serve, and only
    # through an inference profile: the two invocation-side reads the published
    # Mantle entry gets wrong, and the ones a request actually fails on.
    runtime_entry.regions = [_RUNTIME_REGION]
    runtime_entry.inference_profiles = {_RUNTIME_REGION: _RUNTIME_PROFILE}
    runtime = {model_id: runtime_entry}
    _merge_mantle_models(
        runtime, {model_id: _details(model_id, model_id, MANTLE_SERVICE)}
    )
    models._MODELS.clear()  # noqa: SLF001
    models._MODELS.update(runtime)  # noqa: SLF001
    models.update_unified_models_collections()
    try:
        yield model_id
    finally:
        _restore_mantle_registries(saved_registries)
        models._MODELS.clear()  # noqa: SLF001
        models._MODELS.update(saved_models)  # noqa: SLF001
        models.update_unified_models_collections()


@pytest.fixture
def tenant_credential(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve the request under an API key carrying an AWS credential of its own.

    Yields:
        Nothing; the tenant is installed for the duration of the test.
    """
    monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
    token = TENANT.set(
        Tenant(
            key_id="T" + "0" * 15,
            name="tenant",
            aws_credential=TenantAwsCredential(
                role_arn="arn:aws:iam::210987654321:role/stdapi-tenant",
                external_id="external-id-under-test",
            ),
        )
    )
    try:
        yield
    finally:
        TENANT.reset(token)


@pytest.mark.xdist_group("model_cache")
class TestDisplacedRuntimeEntry:
    """The runtime entry a preferred model displaced stays reachable off-catalogue.

    Preferring Mantle for a dual-homed model publishes one entry under one
    identifier, which is what the public catalogue must show. It does not mean
    the model stopped existing on bedrock-runtime: the twin index, the tenant
    service pin and the batch gate all need the runtime entry the merge
    overwrote. Deriving them from the published catalogue alone loses it, and
    every consequence below follows from that one loss.

    Ref: stdapi/models/__init__.py:_merge_mantle_models
         stdapi/models/__init__.py:build_runtime_twins
         docs/operations_authentication_security.md
    """

    def test_the_displaced_entry_is_still_the_runtime_twin(
        self, displaced_runtime_catalog: str
    ) -> None:
        """A dual-homed model pairs with itself when both catalogues name it alike.

        The pairing is derived from the two catalogues, so a model whose Mantle
        and runtime names are the same string pairs with that string. ``None``
        has to keep meaning "no runtime form at all".
        """
        assert models.runtime_twin(displaced_runtime_catalog) == (
            displaced_runtime_catalog
        )

    def test_the_published_catalogue_is_unchanged(
        self, displaced_runtime_catalog: str
    ) -> None:
        """Keeping the runtime entry reachable publishes no second entry.

        The displaced entry is off-catalogue by construction: the model is
        listed once, on the service serving it, under the display name only
        bedrock-runtime carries.
        """
        published = models._ALL_MODELS[displaced_runtime_catalog]  # noqa: SLF001

        assert published.service == MANTLE_SERVICE
        assert published.name == "GPT-5.6 Sol"
        assert len(models._ALL_MODELS) == len(models.EXTRA_MODELS) + 1  # noqa: SLF001

    def test_a_tenant_credential_is_served_by_the_runtime_entry(
        self, displaced_runtime_catalog: str, tenant_credential: None
    ) -> None:
        """A key carrying its own AWS credential reaches the family, not a refusal.

        Mantle rides the deployment's own session, so a tenant-signed request
        is steered onto the runtime entry where its credential signs and pays.
        Refusing instead contradicts what the operations guide promises.
        """
        pinned, pinned_id = models._pin_tenant_billable_service(  # noqa: SLF001
            models._ALL_MODELS[displaced_runtime_catalog],  # noqa: SLF001
            displaced_runtime_catalog,
        )

        assert pinned_id == displaced_runtime_catalog
        assert pinned.service == RUNTIME_SERVICE

    def test_a_tenant_request_is_not_routed_to_mantle(
        self, displaced_runtime_catalog: str, tenant_credential: None
    ) -> None:
        """The pinned model must also pick the runtime chat class.

        Both entries share one identifier, so pinning the catalogue entry
        alone would still resolve the Mantle family and bill the operator --
        the leak the pin exists to close.
        """
        assert not serves_via_mantle(displaced_runtime_catalog)

    def test_a_plain_request_is_still_routed_to_mantle(
        self, displaced_runtime_catalog: str
    ) -> None:
        """Without a tenant credential the preference is what it always was."""
        assert serves_via_mantle(displaced_runtime_catalog)

    def test_a_tenant_request_without_a_runtime_home_is_never_routed_to_mantle(
        self, displaced_runtime_catalog: str, tenant_credential: None
    ) -> None:
        """A lost runtime home must not fall back onto the operator's endpoint.

        The catalogue is rebuilt in the background while requests run, so a
        region outage can leave the Mantle entry published with the runtime
        entry it displaced gone. Picking the Mantle class then signs the call
        with this deployment's own session and bills the operator for someone
        else's request -- the leak the pin exists to close. The runtime class
        is picked instead: with the tenant's credential it fails at AWS, which
        spends nothing on either account.

        Ref: stdapi/models/chat/__init__.py:serves_via_mantle
        """
        models._DISPLACED_RUNTIME_MODELS.clear()  # noqa: SLF001

        assert not serves_via_mantle(displaced_runtime_catalog)

    async def test_a_tenant_request_invokes_the_runtime_regions_and_profile(
        self,
        displaced_runtime_catalog: str,
        tenant_credential: None,
        request_log: dict[str, Any],
    ) -> None:
        """The pin has to reach the reads the invocation itself makes.

        Pinning the catalogue entry and the chat class settles nothing the
        request is sent with: the regions it fails over across and the
        identifier Converse is called with are both read back by model ID,
        which under a shared identifier answers the Mantle entry -- Mantle
        regions, and no inference profile. Sending the bare identifier to a
        region the model only has profile access in is refused by Bedrock.

        Ref: stdapi/models/__init__.py:_invoked_details
        """
        assert await models.compute_candidate_regions(displaced_runtime_catalog) == [
            _RUNTIME_REGION
        ]
        assert (
            await models.resolve_routed_model_id(
                displaced_runtime_catalog, _RUNTIME_REGION, inference_profile=True
            )
            == _RUNTIME_PROFILE
        )

    async def test_a_batch_invokes_the_runtime_regions_and_profile(
        self, displaced_runtime_catalog: str, request_log: dict[str, Any]
    ) -> None:
        """A batch reads the runtime home although no credential says so.

        Batch inference runs on bedrock-runtime alone and no tenant credential
        can submit one, so the binding is stated by the caller rather than
        derived from the request. Without it the job is submitted for a Mantle
        region under the bare identifier and refused a window later.

        Ref: stdapi/batches.py:_resolve_model
        """
        token = models.bind_runtime_home()
        try:
            assert await models.compute_candidate_regions(
                displaced_runtime_catalog
            ) == [_RUNTIME_REGION]
            assert (
                await models.resolve_routed_model_id(
                    displaced_runtime_catalog, _RUNTIME_REGION, inference_profile=True
                )
                == _RUNTIME_PROFILE
            )
            assert get_chat_model(
                displaced_runtime_catalog, allow_mantle=False
            ).model.regions == [_RUNTIME_REGION]
        finally:
            models._RUNTIME_BOUND.reset(token)  # noqa: SLF001

    async def test_a_plain_request_still_invokes_the_published_entry(
        self, displaced_runtime_catalog: str
    ) -> None:
        """Nothing changes for the request the catalogue is published for.

        The displaced entry is read only by a request bedrock-runtime serves;
        a Mantle-served one keeps the regions and the identifier its own
        endpoint answers on.
        """
        assert await models.compute_candidate_regions(displaced_runtime_catalog) == [
            _REGION
        ]

    def test_the_batch_advertisement_is_derived_from_the_runtime_key(
        self, displaced_runtime_catalog: str
    ) -> None:
        """Batch stays unadvertised for the family, on the rate rather than by accident.

        AWS publishes no batch-tier rate for GPT-5.6, so the flag is False --
        as it was while the family was runtime-served. A published rate would
        advertise it, which is what makes the flag derived.
        """
        published = models._ALL_MODELS[displaced_runtime_catalog]  # noqa: SLF001
        priced = frozenset({resolve_model_key(displaced_runtime_catalog)})

        assert models._advertises_batch(published, frozenset()) is False  # noqa: SLF001
        assert models._advertises_batch(published, priced) is True  # noqa: SLF001

    def test_a_batch_runs_the_runtime_chat_class(
        self, displaced_runtime_catalog: str
    ) -> None:
        """A batch names the runtime identifier and runs the runtime family.

        Batches run on bedrock-runtime alone. Now that the model has a runtime
        home the gate stops refusing it upfront, so the class the job is built
        with has to be the runtime one: a Mantle class would translate the
        request for an endpoint the job never reaches.

        Ref: stdapi/batches.py:_batch_model_id
        """
        assert (
            _batch_model_id("gpt-5.6-sol", displaced_runtime_catalog)
            == displaced_runtime_catalog
        )
        resolved = get_chat_model(displaced_runtime_catalog, allow_mantle=False)

        assert not type(resolved).__module__.startswith("stdapi.models.chat._mantle")

    def test_the_displaced_entry_survives_the_published_catalogue(
        self, displaced_runtime_catalog: str
    ) -> None:
        """A server reading the shared catalogue lands in the state a sweep leaves.

        The runtime home is off-catalogue, so it travels beside the catalogue
        rather than inside it. A server restoring without it would refuse the
        family to every credential-carrying key, and batch it under nothing.

        Ref: stdapi/models/_shared_cache.py
        """
        payload = models._catalog_payload(dict(models._MODELS), {})  # noqa: SLF001
        models._DISPLACED_RUNTIME_MODELS.clear()  # noqa: SLF001

        restored = models._restore_catalog(payload, None)  # noqa: SLF001

        assert restored is not None
        assert models.runtime_home(displaced_runtime_catalog) is not None

    def test_restoring_the_registries_is_never_a_re_merge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Putting the catalogue back has to write the snapshot, not re-derive it.

        ``_merge_mantle_models`` records a displaced entry only while the
        bedrock-runtime one is still the entry being overwritten. Re-merging an
        already-merged catalogue -- which is what a saved ``_MODELS`` holds --
        therefore clears ``_DISPLACED_RUNTIME_MODELS`` and rebuilds nothing, so
        a teardown restoring that way leaves every later test on the worker
        with a session catalogue whose runtime homes are gone: the tenant pin
        refuses the family, and the twin index batches it under nothing.

        Ref: stdapi/models/__init__.py:_merge_mantle_models
        """
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_mantle_preferred_models",
            [*DEFAULT_MANTLE_PREFERRED_MODELS],
        )
        model_id = _DUAL_HOMED[0]
        sentinel = {model_id: _details(model_id, "GPT-5.6 Sol", RUNTIME_SERVICE)}
        outer = _snapshot_mantle_registries()
        models._DISPLACED_RUNTIME_MODELS.clear()  # noqa: SLF001
        models._DISPLACED_RUNTIME_MODELS.update(sentinel)  # noqa: SLF001
        try:
            snapshot = _snapshot_mantle_registries()
            merged = {model_id: _details(model_id, "GPT-5.6 Sol", MANTLE_SERVICE)}
            _merge_mantle_models(
                merged, {model_id: _details(model_id, model_id, MANTLE_SERVICE)}
            )
            assert not models._DISPLACED_RUNTIME_MODELS  # noqa: SLF001

            _restore_mantle_registries(snapshot)

            assert sentinel == models._DISPLACED_RUNTIME_MODELS  # noqa: SLF001
        finally:
            _restore_mantle_registries(outer)

    def test_an_endpoint_of_the_operators_is_never_a_runtime_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SageMaker AI or Marketplace endpoint displaced by a Mantle model is not one.

        An operator names its own endpoints, so one can collide with a Mantle
        model identifier. Recording it as the model's runtime home would steer
        every tenant-credentialed request onto a resource the deployment pays
        instance-hours for -- the leak the pin exists to refuse.
        """
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_mantle_preferred_models",
            [*DEFAULT_MANTLE_PREFERRED_MODELS],
        )
        model_id = _DUAL_HOMED[0]
        saved_registries = _snapshot_mantle_registries()
        try:
            endpoint = {model_id: _details(model_id, "GPT-5.6 Sol", SAGEMAKER_SERVICE)}
            _merge_mantle_models(
                endpoint, {model_id: _details(model_id, model_id, MANTLE_SERVICE)}
            )

            assert not models._DISPLACED_RUNTIME_MODELS  # noqa: SLF001
        finally:
            _restore_mantle_registries(saved_registries)


class TestGuardrailRefusal:
    """Guardrails cannot reach a Mantle-served model, so the pair is refused at startup.

    The Mantle invocation path has no ``guardrailConfig`` to attach, and nothing
    at request time can report the guardrail that did not run -- the same reason
    ``aws_bedrock_mantle_service_header`` has been refused since 1.16.

    Ref: stdapi/config.py:_Settings._validate
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
    """

    def test_a_global_guardrail_with_preferred_models_fails_startup(self) -> None:
        """The refusal names the routed models and the way out."""
        with pytest.raises(ValidationError) as excinfo:
            _Settings(
                aws_bedrock_guardrail_identifier="gr-123",
                aws_bedrock_guardrail_version="1",
                aws_bedrock_mantle_preferred_models=["openai.gpt-5.6"],
            )
        message = str(excinfo.value)
        assert "aws_bedrock_mantle_preferred_models" in message
        assert "openai.gpt-5.6" in message
        assert "empty value" in message

    def test_an_alias_guardrail_with_preferred_models_fails_startup(self) -> None:
        """An alias guardrail on a routed model is operator configuration too, so it refuses the same way."""
        aliases: dict[str, Any] = {
            "filtered": {
                "model": "openai.gpt-5.6-sol",
                "guardrail_id": "gr-1",
                "guardrail_version": "1",
            }
        }
        with pytest.raises(ValidationError) as excinfo:
            _Settings(
                aws_bedrock_regions=["us-east-1"],
                model_aliases=aliases,
                aws_bedrock_mantle_preferred_models=["openai.gpt-5.6"],
            )
        assert "aws_bedrock_mantle_preferred_models" in str(excinfo.value)

    def test_an_unrelated_alias_guardrail_boots_with_the_default_preference(
        self,
    ) -> None:
        """An alias guardrail on a model the preference never routes is not refused.

        The alias targets a model outside ``aws_bedrock_mantle_preferred_models``
        (Nova Micro, never GPT-5.6), so it is served by bedrock-runtime under
        its own guardrail regardless of the default preference -- refusing
        startup here would be refusing a combination with no conflict.
        """
        aliases: dict[str, Any] = {
            "filtered": {
                "model": "amazon.nova-micro-v1:0",
                "guardrail_id": "gr-1",
                "guardrail_version": "1",
            }
        }
        settings = _Settings(
            aws_bedrock_regions=["us-east-1"],
            model_aliases=aliases,
            aws_bedrock_mantle_preferred_models=[*DEFAULT_MANTLE_PREFERRED_MODELS],
        )
        assert settings.aws_bedrock_mantle_preferred_models == [
            *DEFAULT_MANTLE_PREFERRED_MODELS
        ]

    def test_no_mantle_region_keeps_the_pair_legal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guardrail boots when no configured region serves Bedrock Mantle at all.

        With no region carrying a Mantle endpoint, the preference routes
        nothing -- the same no-op the disabled-Mantle case already covers.
        The test environment defaults ``aws_bedrock_mantle_regions`` to
        ``us-east-1`` (conftest.py), so it is cleared to let the setting
        derive from ``aws_bedrock_regions`` as an unconfigured deployment would.
        """
        monkeypatch.delenv("aws_bedrock_mantle_regions", raising=False)
        settings = _Settings(
            aws_bedrock_regions=["ca-central-1"],
            aws_bedrock_guardrail_identifier="gr-123",
            aws_bedrock_guardrail_version="1",
            aws_bedrock_mantle_preferred_models=["openai.gpt-5.6"],
        )
        assert settings.aws_bedrock_mantle_regions == []

    def test_clearing_the_preferred_models_accepts_the_guardrail(self) -> None:
        """A guardrailed deployment boots once no model is routed away from it."""
        settings = _Settings(
            aws_bedrock_guardrail_identifier="gr-123",
            aws_bedrock_guardrail_version="1",
            aws_bedrock_mantle_preferred_models=[],
        )
        assert settings.aws_bedrock_guardrail_identifier == "gr-123"

    def test_disabled_mantle_keeps_the_pair_legal(self) -> None:
        """With Mantle off the preference routes nothing, so it is not refused.

        Refusing a no-op would break a deployment whose guardrail is in no
        danger at all.
        """
        settings = _Settings(
            aws_bedrock_guardrail_identifier="gr-123",
            aws_bedrock_guardrail_version="1",
            aws_bedrock_mantle_enabled=False,
            aws_bedrock_mantle_preferred_models=["openai.gpt-5.6"],
        )
        assert settings.aws_bedrock_mantle_preferred_models == ["openai.gpt-5.6"]

    def test_preferred_models_alone_are_accepted(self) -> None:
        """Without a guardrail there is nothing to lose, so the default boots."""
        settings = _Settings(
            aws_bedrock_mantle_preferred_models=[*DEFAULT_MANTLE_PREFERRED_MODELS]
        )
        assert settings.aws_bedrock_mantle_preferred_models == [
            *DEFAULT_MANTLE_PREFERRED_MODELS
        ]


class TestUserRoleIdentityRefusal:
    """The per-end-user role cannot reach a Mantle-served model either.

    A Mantle request is signed from the server's own credentials, so the
    botocore signing hook that assumes ``aws_bedrock_user_role_arn`` -- and the
    ``aws:PrincipalTag`` conditions written on that role -- never runs. Routing
    the family there by default would therefore drop an access-control policy
    the same way it would drop a guardrail, so the pair is refused for the same
    reason and with the same way out.

    Ref: stdapi/config.py:_Settings._validate
         stdapi/aws.py:request_user_role_credentials
         https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html
    """

    def test_requiring_an_identity_with_preferred_models_fails_startup(self) -> None:
        """The refusal names the routed models and the way out."""
        with pytest.raises(ValidationError) as excinfo:
            _Settings(
                aws_bedrock_user_role_arn=_ROLE_ARN,
                aws_bedrock_user_role_require_identity=True,
                aws_bedrock_mantle_preferred_models=["openai.gpt-5.6"],
            )
        message = str(excinfo.value)
        assert "aws_bedrock_user_role_require_identity" in message
        assert "openai.gpt-5.6" in message
        assert "empty value" in message

    def test_clearing_the_preferred_models_accepts_the_requirement(self) -> None:
        """Once no model is routed away from the role, the deployment boots."""
        settings = _Settings(
            aws_bedrock_user_role_arn=_ROLE_ARN,
            aws_bedrock_user_role_require_identity=True,
            aws_bedrock_mantle_preferred_models=[],
        )
        assert settings.aws_bedrock_user_role_require_identity is True

    def test_disabled_mantle_keeps_the_pair_legal(self) -> None:
        """With Mantle off the preference routes nothing, so it is not refused."""
        settings = _Settings(
            aws_bedrock_user_role_arn=_ROLE_ARN,
            aws_bedrock_user_role_require_identity=True,
            aws_bedrock_mantle_enabled=False,
            aws_bedrock_mantle_preferred_models=["openai.gpt-5.6"],
        )
        assert settings.aws_bedrock_mantle_preferred_models == ["openai.gpt-5.6"]

    def test_the_role_alone_is_accepted(self) -> None:
        """Without the requirement there is no policy to lose, so the default boots."""
        settings = _Settings(
            aws_bedrock_user_role_arn=_ROLE_ARN,
            aws_bedrock_mantle_preferred_models=[*DEFAULT_MANTLE_PREFERRED_MODELS],
        )
        assert settings.aws_bedrock_mantle_preferred_models == [
            *DEFAULT_MANTLE_PREFERRED_MODELS
        ]


class TestPriceOfTheMove:
    """Mantle bills the In-Region rate; bedrock-runtime's default routing discounts it.

    Both endpoints are registered from the same model card, so the In-Region
    rate is identical. What the family leaves behind is the Global cross-Region
    profile, which exists on bedrock-runtime alone -- hence a price rise for a
    deployment that changes nothing.

    Ref: stdapi/pricing.py:register_default_prices
    """

    @pytest.mark.parametrize("model_id", _DUAL_HOMED)
    @pytest.mark.parametrize(
        "dimension",
        [
            Dimension.INPUT_TOKENS,
            Dimension.OUTPUT_TOKENS,
            Dimension.CACHE_READ_TOKENS,
            Dimension.CACHE_WRITE_TOKENS,
        ],
    )
    @pytest.mark.parametrize(
        ("in_region_prices", "global_prices"),
        [
            (DEFAULT_MODEL_PRICES, DEFAULT_MODEL_GLOBAL_PRICES),
            (
                DEFAULT_MODEL_LONG_CONTEXT_PRICES,
                DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES,
            ),
        ],
        ids=["short_context", "long_context"],
    )
    def test_mantle_costs_exactly_ten_percent_more_than_global_routing(
        self,
        model_id: str,
        dimension: Dimension,
        in_region_prices: dict[str, dict[Dimension, str]],
        global_prices: dict[str, dict[Dimension, str]],
    ) -> None:
        """Every billed token dimension is 1.1x the rate the Global profile charges.

        The short and long-context tables both carry the ratio the operator
        documentation commits to: a prompt past
        ``MODEL_LONG_CONTEXT_THRESHOLDS`` bills the whole call at the long rates,
        so a Global long rate republished at another ratio would move the price
        of the default routing with nothing else noticing.
        """
        in_region = Decimal(in_region_prices[model_id][dimension])
        global_routed = Decimal(global_prices[model_id][dimension])
        assert in_region == global_routed * Decimal("1.1")

    @pytest.mark.parametrize("model_id", _DUAL_HOMED)
    def test_the_catalogue_prices_mantle_in_region_and_never_global(
        self, model_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loaded catalogue resolves Mantle at In-Region, with no Global row to relax onto.

        ``resolve_price`` relaxes the routing axis, so a Mantle call asking for
        Global routing must still land on the In-Region rate rather than on the
        discount bedrock-runtime gets.
        """
        index: dict[PriceKey, Price] = {}
        _apply_default_prices(index)
        monkeypatch.setattr(_state, "price_index", index)

        in_region = Decimal(DEFAULT_MODEL_PRICES[model_id][Dimension.INPUT_TOKENS])
        mantle = resolve_price(
            Service.BEDROCK_MANTLE, model_id, _REGION, Dimension.INPUT_TOKENS
        )
        mantle_global = resolve_price(
            Service.BEDROCK_MANTLE,
            model_id,
            _REGION,
            Dimension.INPUT_TOKENS,
            routing="global",
        )
        runtime_global = resolve_price(
            Service.BEDROCK, model_id, _REGION, Dimension.INPUT_TOKENS, routing="global"
        )
        assert mantle is not None
        assert mantle_global is not None
        assert runtime_global is not None
        assert mantle.amount == in_region
        assert mantle_global.amount == in_region
        assert runtime_global.amount * Decimal("1.1") == in_region

    @pytest.mark.parametrize("model_id", _DUAL_HOMED)
    def test_the_catalogue_prices_a_long_context_mantle_call_in_region(
        self, model_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt past the long-context boundary keeps the In-Region long rate.

        ``resolve_price`` relaxes the routing axis on the long-context rows too,
        so a Mantle call asking for Global routing must not land on the Global
        long rate bedrock-runtime gets -- the same 10% the short rates carry.
        """
        index: dict[PriceKey, Price] = {}
        _apply_default_prices(index)
        monkeypatch.setattr(_state, "price_index", index)

        long_in_region = Decimal(
            DEFAULT_MODEL_LONG_CONTEXT_PRICES[model_id][Dimension.INPUT_TOKENS]
        )
        mantle_long = resolve_price(
            Service.BEDROCK_MANTLE,
            model_id,
            _REGION,
            Dimension.INPUT_TOKENS,
            routing="global",
            context="long",
        )
        runtime_long_global = resolve_price(
            Service.BEDROCK,
            model_id,
            _REGION,
            Dimension.INPUT_TOKENS,
            routing="global",
            context="long",
        )
        assert mantle_long is not None
        assert runtime_long_global is not None
        assert mantle_long.amount == long_in_region
        assert runtime_long_global.amount * Decimal("1.1") == long_in_region

    def test_the_operator_reference_quotes_the_rates_the_gateway_charges(self) -> None:
        """The one prose copy of the per-million figures matches the tables above.

        A repriced model moves the tables and nothing else, so the sentence an
        operator sizes spend from is compared to them rather than trusted --
        the figures are quoted here and nowhere else for that reason.

        Ref: docs/operations_configuration.md#bedrock-mantle-preferred-models
        """
        sentences = [
            line
            for line in _PRICE_REFERENCE.read_text(encoding="utf-8").splitlines()
            if _PRICE_SENTENCE_MARKER in line
        ]
        assert len(sentences) == 1, sentences

        assert set(findall(r"\$\d+\.\d\d", sentences[0])) == {
            f"${Decimal(table[model_id][dimension]) * 1_000_000:.2f}"
            for table in (DEFAULT_MODEL_PRICES, DEFAULT_MODEL_GLOBAL_PRICES)
            for model_id in _DUAL_HOMED
            for dimension in (Dimension.INPUT_TOKENS, Dimension.OUTPUT_TOKENS)
        }
