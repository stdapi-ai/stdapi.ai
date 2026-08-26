"""The model catalog cache: serving an expired catalog, and sharing a fresh one.

Two halves, gated differently and tested together because they share one state
machine. **Stale-serving** ships for every deployment: once the catalog reaches
``model_cache_seconds`` the request that notices is answered from the catalog in
hand while the refresh runs behind it, and only a catalog that has never been
built or has passed ``model_cache_max_stale_seconds`` makes a caller wait. The
**shared catalog** is opt-in and needs a DynamoDB table: one server sweeps and
publishes, the rest read.

The state machine is the risk, so it is what is asserted: that exactly one sweep
runs however many callers arrive, that a failure is reported and backed off
rather than raised at a caller, that the ceiling really does force a wait, and
that a background refresh cannot outlive the shutdown drain. The shared half is
exercised against a local DynamoDB stand-in; what only the real service can
settle stays in ``tests/test_aws_dynamodb.py``.

Ref: stdapi/models/__init__.py:initialize_bedrock_models
     stdapi/models/_shared_cache.py
     stdapi/config.py:_Settings.model_cache_max_stale_seconds
"""

from __future__ import annotations

from asyncio import Event, gather, sleep
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import EndpointConnectionError

import stdapi.models
from stdapi import server
from stdapi.aws_dynamodb import (
    EXPIRES_AT_ATTRIBUTE,
    PARTITION_KEY,
    SCHEMA_ATTRIBUTE,
    SCHEMA_VERSION,
    SORT_KEY,
    get_item,
    item_key,
    put_item,
)
from stdapi.config import SETTINGS
from stdapi.models import (
    MANTLE_MODELS,
    MARKETPLACE_ENDPOINT_MODELS,
    _shared_cache,
    catalog_generation,
    drain_model_refresh,
    initialize_bedrock_models,
    validate_model,
)
from stdapi.utils import to_json_bytes
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from stdapi.models import ModelDetails

#: Every test here drives the in-process catalog directly.
pytestmark = [
    pytest.mark.local,
    # The catalog is a module singleton and the shared records sit under fixed
    # keys, so readers and writers alike have to run on one worker.
    pytest.mark.xdist_group("model_cache"),
]

#: Module state a refresh rewrites, saved and restored around every test.
_CATALOG_COLLECTIONS = (
    "_MODELS",
    "_ALL_MODELS",
    "_MODELS_INPUT_MODALITY",
    "_MODELS_OUTPUT_MODALITY",
    "_ALL_MODELS_INPUT_MODALITY",
    "_ALL_MODELS_OUTPUT_MODALITY",
    "MANTLE_MODELS",
    "MARKETPLACE_ENDPOINT_MODELS",
    "MODEL_ALIASES",
)


class _Sweep:
    """A stand-in for the multi-region discovery sweep, counting its calls."""

    def __init__(
        self, models: dict[str, ModelDetails], error: Exception | None = None
    ) -> None:
        """Record what the sweep answers, and how.

        Args:
            models: The catalog it returns.
            error: Raised instead of answering, when given.
        """
        self.models = models
        self.error = error
        self.calls = 0

    async def __call__(
        self,
        failed_regions: dict[str, str],
        mantle_regions_without_endpoint: dict[str, str],
        unavailable_models: dict[str, dict[str, list[str]]],
    ) -> tuple[dict[str, ModelDetails], dict[str, str]]:
        """Answer as ``_collect_all_models`` does.

        Args:
            failed_regions: Accumulator, unused.
            mantle_regions_without_endpoint: Accumulator, unused.
            unavailable_models: Accumulator, unused.

        Returns:
            The catalog and no invalid ARN mappings.

        Raises:
            Exception: Whatever this sweep was built to fail with.
        """
        del failed_regions, mantle_regions_without_endpoint, unavailable_models
        self.calls += 1
        # Yields the loop, so a concurrent caller can reach the single-flight
        # guard while this one is still "sweeping".
        await sleep(0)
        if self.error is not None:
            raise self.error
        return dict(self.models), {}


@pytest.fixture
async def catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., _Sweep]]:
    """Isolate the model catalog and hand out a stand-in discovery sweep.

    The catalog, the collections a refresh rewrites and the cache state are all
    module singletons, so each test gets them back exactly as it found them.

    Yields:
        A factory installing a sweep and returning it for its call count.

    Ref: stdapi/models/__init__.py:_CACHE
    """
    saved = {name: dict(getattr(stdapi.models, name)) for name in _CATALOG_COLLECTIONS}
    # Emptied as well as saved: whatever the rest of the suite left in the
    # catalog would otherwise be published, restored and counted here.
    for name in _CATALOG_COLLECTIONS:
        getattr(stdapi.models, name).clear()
    for key in ("update_next", "updated_at", "refresh_task"):
        monkeypatch.setitem(stdapi.models._CACHE, key, None)  # noqa: SLF001
    monkeypatch.setattr(SETTINGS, "model_cache_shared", False)
    # A refresh that finds new models otherwise reloads real prices from AWS.
    monkeypatch.setattr(
        stdapi.models, "refresh_price_catalog_for_new_models", _no_price_reload
    )
    monkeypatch.setattr(_shared_cache, "_REPORTED", set())

    def _install(*model_ids: str, error: Exception | None = None) -> _Sweep:
        sweep = _Sweep(
            {model_id: make_model_details(model_id) for model_id in model_ids}, error
        )
        monkeypatch.setattr(stdapi.models, "_collect_all_models", sweep)
        return sweep

    try:
        yield _install
    finally:
        await drain_model_refresh(5.0)
        for name, content in saved.items():
            target = getattr(stdapi.models, name)
            target.clear()
            target.update(content)


async def _no_price_reload(_new_model_ids: set[str]) -> None:
    """Stand in for the price-catalog reload a new model triggers."""


def _age_catalog(seconds: int, *, since_success: int | None = None) -> None:
    """Backdate the catalog so it reads as expired.

    Args:
        seconds: How long ago the refresh deadline passed.
        since_success: How long ago the last successful refresh was; defaults
            to *seconds* plus the refresh interval.
    """
    now = datetime.now(UTC)
    interval = stdapi.models._CACHE["update_interval"]  # noqa: SLF001
    stdapi.models._CACHE["update_next"] = now - timedelta(seconds=seconds)  # noqa: SLF001
    stdapi.models._CACHE["updated_at"] = (  # noqa: SLF001
        now - timedelta(seconds=since_success)
        if since_success is not None
        else now - timedelta(seconds=seconds) - interval
    )


class TestServingAnExpiredCatalog:
    """The state machine deciding who waits for a refresh and who does not.

    Ref: stdapi/models/__init__.py:initialize_bedrock_models
    """

    async def test_a_current_catalog_costs_nothing(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """A catalog inside its lifetime is not refreshed and no one waits.

        Ref: stdapi/models/__init__.py:_refresh_due
        """
        sweep = catalog("vendor.one")
        await initialize_bedrock_models()
        assert sweep.calls == 1

        assert await initialize_bedrock_models() is False
        assert sweep.calls == 1

    async def test_a_cold_catalog_is_awaited(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """With nothing to serve, the caller waits for the sweep.

        Ref: stdapi/models/__init__.py:_may_serve_stale
        """
        sweep = catalog("vendor.one")

        assert await initialize_bedrock_models() is True

        assert sweep.calls == 1
        assert "vendor.one" in stdapi.models._MODELS  # noqa: SLF001

    async def test_an_expired_catalog_answers_while_it_refreshes(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """The expired catalog is served immediately; the sweep runs behind it.

        Ref: stdapi/models/__init__.py:_schedule_refresh
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60)
        sweep = catalog("vendor.one", "vendor.two")

        assert await initialize_bedrock_models() is False
        # The refresh has not run yet, and the old catalog is what answers.
        assert sweep.calls == 0
        assert "vendor.two" not in stdapi.models._MODELS  # noqa: SLF001

        await drain_model_refresh(5.0)

        assert sweep.calls == 1
        assert "vendor.two" in stdapi.models._MODELS  # noqa: SLF001

    async def test_concurrent_callers_trigger_one_refresh(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """However many requests notice the expiry, exactly one sweep runs.

        Ref: stdapi/models/__init__.py:_schedule_refresh
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60)
        sweep = catalog("vendor.one", "vendor.two")

        await gather(*(initialize_bedrock_models() for _ in range(8)))
        await drain_model_refresh(5.0)

        assert sweep.calls == 1

    async def test_the_ceiling_makes_the_caller_wait_again(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """Past model_cache_max_stale_seconds the refresh becomes blocking again.

        This is what stops a deployment whose refreshes fail forever from
        advertising a withdrawn model for the rest of its life.

        Ref: stdapi/config.py:_Settings.model_cache_max_stale_seconds
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60, since_success=SETTINGS.model_cache_max_stale_seconds + 60)
        sweep = catalog("vendor.two")

        assert await initialize_bedrock_models() is True

        assert sweep.calls == 1
        assert "vendor.one" not in stdapi.models._MODELS  # noqa: SLF001

    async def test_a_zero_ceiling_never_serves_an_expired_catalog(
        self, catalog: Callable[..., _Sweep], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model_cache_max_stale_seconds=0 restores the fully synchronous refresh.

        Ref: stdapi/config.py:_Settings.model_cache_max_stale_seconds
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        monkeypatch.setitem(
            stdapi.models._CACHE,  # noqa: SLF001
            "max_stale",
            timedelta(0),
        )
        _age_catalog(1, since_success=1)
        sweep = catalog("vendor.two")

        assert await initialize_bedrock_models() is True

        assert sweep.calls == 1

    async def test_a_failed_background_refresh_reaches_no_caller(
        self, catalog: Callable[..., _Sweep], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A refresh that fails is reported to the operator and backed off.

        The caller that triggered it already has its answer, so the failure has
        nowhere to be raised -- and the next request must not immediately try
        again. It is reported at ``warning``, which with the deeply-stale case
        below brackets the escalation boundary: every transient Bedrock hiccup
        would otherwise page an operator.

        Ref: stdapi/models/__init__.py:_refresh_in_background
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        # Expired, but within the intervals that keep this a warning.
        _age_catalog(1, since_success=SETTINGS.model_cache_seconds + 1)
        sweep = catalog(error=EndpointConnectionError(endpoint_url="https://x.invalid"))

        assert await initialize_bedrock_models() is False
        await drain_model_refresh(5.0)

        assert sweep.calls == 1
        assert "vendor.one" in stdapi.models._MODELS  # noqa: SLF001
        reported = capsys.readouterr().out
        assert "model_cache_refresh" in reported
        assert '"level":"warning"' in reported
        assert '"level":"error"' not in reported
        # Backed off: the next request serves the same catalog without sweeping.
        assert await initialize_bedrock_models() is False
        assert sweep.calls == 1

    async def test_a_deeply_stale_failure_is_reported_as_an_error(
        self, catalog: Callable[..., _Sweep], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Past two refresh intervals a failing refresh is louder than a warning.

        Past ``_DEGRADED_INTERVALS`` intervals and no sooner: the warning case
        above is the other side of the same boundary.

        Ref: stdapi/models/__init__.py:_refresh_in_background
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60, since_success=SETTINGS.model_cache_seconds * 3)
        catalog(error=EndpointConnectionError(endpoint_url="https://x.invalid"))

        await initialize_bedrock_models()
        await drain_model_refresh(5.0)

        assert '"level":"error"' in capsys.readouterr().out

    async def test_the_refresh_does_not_outlive_the_shutdown_drain(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """The background refresh is drained with the rest of the deferred work.

        Ref: stdapi/models/__init__.py:drain_model_refresh
             stdapi/cleanup.py:drain_tasks
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60)
        sweep = catalog("vendor.one", "vendor.two")
        await initialize_bedrock_models()

        assert await drain_model_refresh(5.0) == 0

        assert sweep.calls == 1
        assert not stdapi.models._REFRESH_TASKS  # noqa: SLF001

    async def test_a_refresh_past_the_deadline_is_cancelled_and_counted(
        self, catalog: Callable[..., _Sweep], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sweep that will not finish in time is dropped, not waited out.

        The count is what the stop event reports, under the registry name the
        model refresh occupies: a shutdown that hangs on a multi-region sweep
        is the failure this deadline exists to prevent.

        Ref: stdapi/models/__init__.py:drain_model_refresh
             stdapi/main.py:drain_background_tasks
        """
        from stdapi.main import drain_background_tasks  # noqa: PLC0415

        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60)
        # A sweep that never returns, so the deadline is what ends it.
        never_finishes = Event()

        async def _blocked(*_accumulators: object) -> None:
            await never_finishes.wait()

        monkeypatch.setattr(stdapi.models, "_collect_all_models", _blocked)
        monkeypatch.setattr(SETTINGS, "shutdown_drain_timeout", 0.05)

        await initialize_bedrock_models()
        refresh = stdapi.models._CACHE["refresh_task"]  # noqa: SLF001

        assert await drain_background_tasks() == {"model_refresh": 1}

        assert refresh is not None
        assert refresh.cancelled()


class TestCatalogGeneration:
    """What tells a listing route its cached payload is out of date.

    A refresh that completes in the background answers nobody, so the flag the
    routes used to rebuild from is not enough on its own.

    Ref: stdapi/models/__init__.py:catalog_generation
    """

    async def test_a_changed_catalog_advances_the_generation(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """Every catalog change is visible to a route that did not trigger it.

        Ref: stdapi/routes/openai_models.py:list_models
        """
        catalog("vendor.one")
        await initialize_bedrock_models()
        before = catalog_generation()
        _age_catalog(60)
        catalog("vendor.one", "vendor.two")

        await initialize_bedrock_models()
        await drain_model_refresh(5.0)

        assert catalog_generation() > before

    async def test_an_unchanged_catalog_leaves_it_alone(
        self, catalog: Callable[..., _Sweep]
    ) -> None:
        """A refresh finding the same models rebuilds nothing.

        Ref: stdapi/models/__init__.py:_install_catalog
        """
        sweep = catalog("vendor.one")
        await initialize_bedrock_models()
        before = catalog_generation()
        _age_catalog(60)

        await initialize_bedrock_models()
        await drain_model_refresh(5.0)

        assert sweep.calls == 2
        assert catalog_generation() == before


class TestWithdrawnModels:
    """What bounds how long a model AWS has taken away stays on offer.

    Ref: stdapi/models/__init__.py:validate_model
    """

    async def test_resolving_a_model_starts_the_refresh_that_removes_it(
        self, catalog: Callable[..., _Sweep], request_log: dict[str, Any]
    ) -> None:
        """A resolution against an expired catalog heals it, without waiting.

        This is what keeps the exposure at one refresh rather than one full
        cache lifetime: the first request naming any model starts the sweep.

        Ref: stdapi/models/__init__.py:validate_model
        """
        del request_log
        catalog("vendor.one", "vendor.two")
        await initialize_bedrock_models()
        _age_catalog(60)
        sweep = catalog("vendor.one")

        # Still served from the expired catalog, which is the documented window.
        assert (await validate_model("vendor.two")).id == "vendor.two"

        await drain_model_refresh(5.0)

        assert sweep.calls == 1
        assert "vendor.two" not in stdapi.models._MODELS  # noqa: SLF001

    async def test_an_unknown_model_waits_for_the_refresh(
        self, catalog: Callable[..., _Sweep], request_log: dict[str, Any]
    ) -> None:
        """A model the catalog does not know is worth waiting a refresh for.

        Answering "no such model" from a catalog too old to have heard of it
        would make a newly released model unusable for a whole lifetime.

        Ref: stdapi/models/__init__.py:validate_model
        """
        del request_log
        catalog("vendor.one")
        await initialize_bedrock_models()
        _age_catalog(60)
        catalog("vendor.one", "vendor.new")

        assert (await validate_model("vendor.new")).id == "vendor.new"


class TestSharedCatalogIsOptional:
    """The DynamoDB half stays inert until it is asked for.

    Ref: stdapi/models/_shared_cache.py:enabled
    """

    def test_it_is_off_without_the_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment that did not ask for it makes no table call.

        Ref: stdapi/models/_shared_cache.py:enabled
        """
        monkeypatch.setattr(SETTINGS, "model_cache_shared", False)
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "some-table")

        assert _shared_cache.enabled() is False

    def test_it_is_off_without_a_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asking for it without a table leaves it off rather than failing calls.

        Configuration refuses that combination at startup; this is the belt to
        that braces, so a monkeypatched or reloaded setting cannot make the
        module call a table it has no name for.

        Ref: stdapi/models/_shared_cache.py:enabled
             stdapi/config.py:_Settings._validate_dynamodb
        """
        monkeypatch.setattr(SETTINGS, "model_cache_shared", True)
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", None)

        assert _shared_cache.enabled() is False

    def test_the_settings_must_agree(self) -> None:
        """model_cache_shared without a table is refused at startup, by name.

        Ref: stdapi/config.py:_Settings._validate_dynamodb
        """
        from pydantic import ValidationError  # noqa: PLC0415

        from stdapi.config import _Settings  # noqa: PLC0415

        with pytest.raises(ValidationError, match="model_cache_shared requires"):
            _Settings(model_cache_shared=True, aws_dynamodb_table=None)


class TestSharedCatalog:
    """Publishing a catalog and reading it back, against a local stand-in.

    Ref: stdapi/models/_shared_cache.py
    """

    @staticmethod
    def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
        """Turn the shared catalog on for a test already bound to a table."""
        monkeypatch.setattr(SETTINGS, "model_cache_shared", True)

    @staticmethod
    def _reported(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Collect what the shared catalog reports to the operator.

        Args:
            monkeypatch: The test's patcher.

        Returns:
            The list the warnings are appended to as they are reported.
        """
        reported: list[str] = []
        monkeypatch.setattr(_shared_cache, "_REPORTED", set())
        monkeypatch.setattr(
            _shared_cache,
            "log_error_details",
            lambda detail, level: reported.append(detail),  # noqa: ARG005
        )
        return reported

    @staticmethod
    async def _publish_raw(blob: bytes, checksum: str, shard_count: int = 1) -> None:
        """Write a manifest and one shard holding exactly *blob*.

        Bypasses ``publish_catalog`` so a test can put in the table what only a
        defect or another writer would ever produce.

        Args:
            blob: The compressed bytes to store as the single shard.
            checksum: The digest the manifest claims for them.
            shard_count: The piece count the manifest claims.
        """
        partition = item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint())
        await put_item(
            {
                PARTITION_KEY: partition,
                SORT_KEY: item_key(_shared_cache._SHARD, "cafe", "0000"),  # noqa: SLF001
                "data": blob,
            }
        )
        await put_item(
            {
                PARTITION_KEY: partition,
                SORT_KEY: "manifest",
                "version": "cafe",
                "created_at": int(datetime.now(UTC).timestamp()),
                "shard_count": shard_count,
                "checksum": checksum,
            }
        )

    async def test_a_second_server_reads_instead_of_sweeping(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What one server published is what the next one starts from.

        Ref: stdapi/models/_shared_cache.py:publish_catalog
             stdapi/models/_shared_cache.py:read_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        publisher = catalog("vendor.one", "vendor.two")
        await initialize_bedrock_models()
        assert publisher.calls == 1

        # A second server: same fingerprint, empty catalog of its own.
        monkeypatch.setitem(stdapi.models._CACHE, "update_next", None)  # noqa: SLF001
        monkeypatch.setitem(stdapi.models._CACHE, "updated_at", None)  # noqa: SLF001
        stdapi.models._MODELS.clear()  # noqa: SLF001
        reader = catalog("vendor.three")

        await initialize_bedrock_models()

        assert reader.calls == 0
        assert set(stdapi.models._MODELS) == {"vendor.one", "vendor.two"}  # noqa: SLF001

    async def test_the_private_routing_fields_survive_the_round_trip(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reader gets the state it needs to invoke, not just to advertise.

        The per-region profile cache and the Marketplace endpoint ARNs are kept
        out of the public model representation, so a plain dump would drop them
        and leave the reader unable to route a request.

        Ref: stdapi/models/__init__.py:_dump_models
        """
        del dynamodb_table
        self._enable(monkeypatch)
        model = make_model_details(
            "vendor.one",
            inference_profiles={"us-east-1": "us.vendor.one"},
            inference_profiles_regional={"us-east-1": "us.vendor.one"},
            marketplace_endpoints={"us-east-1": "arn:aws:sagemaker:::endpoint/x"},
        )
        # The collections a sweep fills as a side effect: a reader that did not
        # sweep has to end up holding them too, or it advertises a model it
        # cannot route.
        MANTLE_MODELS["vendor.mantle"] = make_model_details("vendor.mantle")
        MARKETPLACE_ENDPOINT_MODELS["vendor.one"] = model
        payload = stdapi.models._catalog_payload({"vendor.one": model}, {})  # noqa: SLF001
        MANTLE_MODELS.clear()
        MARKETPLACE_ENDPOINT_MODELS.clear()
        await _shared_cache.publish_catalog(
            payload, int(datetime.now(UTC).timestamp()), None
        )

        published = await _shared_cache.read_catalog(None)
        assert published is not None
        restored = stdapi.models._restore_catalog(published.payload, None)  # noqa: SLF001

        assert restored is not None
        rebuilt = restored[0]["vendor.one"]
        assert rebuilt.inference_profiles == {"us-east-1": "us.vendor.one"}
        assert rebuilt.inference_profiles_regional == {"us-east-1": "us.vendor.one"}
        assert rebuilt.marketplace_endpoints == {
            "us-east-1": "arn:aws:sagemaker:::endpoint/x"
        }
        assert set(MANTLE_MODELS) == {"vendor.mantle"}
        assert set(MARKETPLACE_ENDPOINT_MODELS) == {"vendor.one"}
        assert MARKETPLACE_ENDPOINT_MODELS["vendor.one"].marketplace_endpoints == {
            "us-east-1": "arn:aws:sagemaker:::endpoint/x"
        }

    async def test_a_reader_inherits_the_age_of_what_it_read(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A read catalog expires when its publisher's would, not a full interval later.

        Dating it from the read instead would let each server restart the whole
        ``model_cache_seconds`` from whenever it happened to start, so a fleet
        could hold a catalogue approaching twice that age and a rolling
        scale-out would never re-sweep in step.

        Ref: stdapi/models/__init__.py:_collect_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        # Published in the past, but still inside model_cache_seconds, so the
        # difference between its age and the read time is what is asserted.
        elapsed = max(1, SETTINGS.model_cache_seconds // 2)
        published_at = int(datetime.now(UTC).timestamp()) - elapsed
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        await _shared_cache.publish_catalog(payload, published_at, None)
        reader = catalog("vendor.two")

        await initialize_bedrock_models()

        assert reader.calls == 0
        expected = datetime.fromtimestamp(published_at, tz=UTC)
        assert stdapi.models._CACHE["updated_at"] == expected  # noqa: SLF001
        assert stdapi.models._CACHE["update_next"] == (  # noqa: SLF001
            expected + stdapi.models._CACHE["update_interval"]  # noqa: SLF001
        )

    async def test_a_catalog_too_big_for_one_item_is_cut_up(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A catalog past DynamoDB's item limit round-trips across several items.

        Ref: stdapi/models/_shared_cache.py:publish_catalog
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ServiceQuotas.html
        """
        del catalog, dynamodb_table
        self._enable(monkeypatch)
        # Small enough to publish, large enough that zstd cannot fit it in one
        # item: random-ish names do not compress away.
        monkeypatch.setattr(_shared_cache, "_SHARD_BYTES", 512)
        models = {
            f"vendor.model-{index:04d}": make_model_details(f"vendor.model-{index:04d}")
            for index in range(400)
        }
        payload = stdapi.models._catalog_payload(models, {})  # noqa: SLF001

        await _shared_cache.publish_catalog(
            payload, int(datetime.now(UTC).timestamp()), None
        )
        published = await _shared_cache.read_catalog(None)

        manifest = await get_item(
            item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint()), "manifest"
        )
        assert manifest is not None
        assert isinstance(manifest["shard_count"], int)
        assert manifest["shard_count"] > 1, "the catalog was not cut up at all"
        assert published is not None
        restored = stdapi.models._restore_catalog(published.payload, None)  # noqa: SLF001
        assert restored is not None
        assert set(restored[0]) == set(models)

    async def test_a_catalog_that_will_not_fit_is_not_published(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Past the shard ceiling the operator is told rather than the table filled.

        Ref: stdapi/models/_shared_cache.py:publish_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        monkeypatch.setattr(_shared_cache, "_SHARD_BYTES", 16)
        monkeypatch.setattr(_shared_cache, "_MAX_SHARDS", 2)
        reported = self._reported(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {
                f"vendor.m{index}": make_model_details(f"vendor.m{index}")
                for index in range(20)
            },
            {},
        )

        await _shared_cache.publish_catalog(
            payload, int(datetime.now(UTC).timestamp()), None
        )

        assert await _shared_cache.read_catalog(None) is None
        # Silence here would leave an operator with sharing switched on, a
        # healthy table and every server sweeping for itself forever.
        assert len(reported) == 1
        assert "model_cache_shared" in reported[0]
        assert "was not shared" in reported[0]

    async def test_a_torn_write_reads_as_no_catalog_at_all(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest naming pieces that are not all there is a miss, never a partial read.

        Ref: stdapi/models/_shared_cache.py:_read_shards
        """
        del dynamodb_table
        self._enable(monkeypatch)
        await put_item(
            {
                PARTITION_KEY: item_key(
                    _shared_cache.NAMESPACE, _shared_cache.fingerprint()
                ),
                SORT_KEY: "manifest",
                "version": "deadbeef",
                "created_at": int(datetime.now(UTC).timestamp()),
                "shard_count": 3,
                "checksum": "0" * 64,
            }
        )

        assert await _shared_cache.read_catalog(None) is None

    async def test_a_manifest_from_the_future_is_a_miss_not_a_freeze(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A created_at ahead of this clock is refused rather than served forever.

        Unbounded, a future ``created_at`` would set ``update_next`` in the
        future too, so the staleness ceiling would never fire and the fleet
        would serve that record until restart.

        Ref: stdapi/models/_shared_cache.py:read_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        future = int(datetime.now(UTC).timestamp()) + 3600
        await _shared_cache.publish_catalog(payload, future, None)

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 1
        assert "created_at" in reported[0]

    async def test_an_out_of_range_created_at_is_a_miss_not_a_crash(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A created_at outside datetime's range degrades to a local sweep.

        A value ``datetime.fromtimestamp`` cannot represent (a wire-decoded int
        outside its range) must not escape as an unhandled exception: that would
        crash every reader's refresh and, at startup, fail the boot for every
        server sharing the fingerprint.

        Ref: stdapi/models/_shared_cache.py:read_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        await _shared_cache.publish_catalog(payload, 10**18, None)

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 1
        assert "created_at" in reported[0]

    async def test_a_shard_that_was_tampered_with_fails_its_checksum(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A complete shard set whose bytes changed is a miss and a warning.

        The piece count adding up is not enough: a shard rewritten in place
        leaves the manifest pointing at a set that decodes to something its
        publisher never wrote, and the checksum is what catches it.

        Ref: stdapi/models/_shared_cache.py:_read_shards
        """
        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        await _shared_cache.publish_catalog(
            payload, int(datetime.now(UTC).timestamp()), None
        )
        manifest = await get_item(
            item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint()), "manifest"
        )
        assert manifest is not None
        await put_item(
            {
                PARTITION_KEY: item_key(
                    _shared_cache.NAMESPACE, _shared_cache.fingerprint()
                ),
                SORT_KEY: item_key(
                    _shared_cache._SHARD,  # noqa: SLF001
                    str(manifest["version"]),
                    "0000",
                ),
                "data": b"not what was published",
            }
        )

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 1
        assert "does not match its own checksum" in reported[0]

    async def test_a_blob_that_does_not_decode_is_a_miss(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A frame agreeing with its own checksum is still refused if it will not decode.

        The checksum only proves the pieces are the ones the manifest names, so
        the decode is the last thing between the table and the catalog served,
        and it fails in two ways: a frame that never ends, and one that ends on
        something that is not the published list.

        Ref: stdapi/models/_shared_cache.py:_read_shards
        """
        from compression.zstd import compress  # noqa: PLC0415

        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        truncated = compress(to_json_bytes(payload))[:-8]

        await self._publish_raw(truncated, sha256(truncated).hexdigest())

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 1
        assert "stops short of its own end" in reported[0]

        not_json = compress(b"{ this was never a model list")
        await self._publish_raw(not_json, sha256(not_json).hexdigest())

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 2
        assert "could not be decoded" in reported[1]

    async def test_a_record_from_a_newer_build_is_skipped(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A layout this build does not know is read as an empty cache, silently.

        A rolling deployment puts both builds on the same table, and a newer
        one's records have to be skipped rather than half-understood.

        Ref: stdapi/models/_shared_cache.py:read_catalog
             stdapi/aws_dynamodb.py:readable_schema
        """
        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        await _shared_cache.publish_catalog(
            payload, int(datetime.now(UTC).timestamp()), None
        )
        partition = item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint())
        manifest = await get_item(partition, "manifest")
        assert manifest is not None
        assert await _shared_cache.read_catalog(None) is not None, "not readable at all"

        await put_item(dict(manifest) | {SCHEMA_ATTRIBUTE: SCHEMA_VERSION + 1})

        assert await _shared_cache.read_catalog(None) is None
        assert reported == [], "a newer build's record is not a fault to report"

    async def test_a_shard_from_a_newer_build_is_skipped(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The layout gate covers the pieces as well as the manifest naming them.

        Ref: stdapi/models/_shared_cache.py:_read_shards
             stdapi/aws_dynamodb.py:readable_schema
        """
        del dynamodb_table
        self._enable(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        await _shared_cache.publish_catalog(
            payload, int(datetime.now(UTC).timestamp()), None
        )
        partition = item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint())
        manifest = await get_item(partition, "manifest")
        assert manifest is not None
        shard_key = item_key(
            _shared_cache._SHARD,  # noqa: SLF001
            str(manifest["version"]),
            "0000",
        )
        shard = await get_item(partition, shard_key)
        assert shard is not None

        await put_item(dict(shard) | {SCHEMA_ATTRIBUTE: SCHEMA_VERSION + 1})

        assert await _shared_cache.read_catalog(None) is None

    async def test_a_manifest_naming_more_pieces_than_exist_is_refused(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A piece count past the publishing ceiling is rejected before it is read.

        Nothing this build publishes spans more than ``_MAX_SHARDS`` pieces, so
        a manifest that claims to is a record to refuse rather than a query to
        pay for and a buffer to fill.

        Ref: stdapi/models/_shared_cache.py:read_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)

        await self._publish_raw(
            b"unread",
            sha256(b"unread").hexdigest(),
            _shared_cache._MAX_SHARDS + 1,  # noqa: SLF001
        )

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 1
        assert "pieces" in reported[0]

    async def test_a_blob_that_expands_past_the_ceiling_is_refused(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decompression is bounded, so a small item cannot claim the server's memory.

        The blob is expanded before anything has established where it came
        from, and zstd reaches ratios that turn a few hundred kilobytes of
        table content into gigabytes of allocation on every server at once.

        Ref: stdapi/models/_shared_cache.py:_read_shards
        """
        from compression.zstd import compress  # noqa: PLC0415

        del dynamodb_table
        self._enable(monkeypatch)
        reported = self._reported(monkeypatch)
        monkeypatch.setattr(_shared_cache, "_MAX_PAYLOAD_BYTES", 1024)
        bomb = compress(b'{"models":' + b" " * 1_000_000 + b"}")
        assert len(bomb) < 1024, "the blob has to be smaller than what it expands to"

        await self._publish_raw(bomb, sha256(bomb).hexdigest())

        assert await _shared_cache.read_catalog(None) is None
        assert len(reported) == 1
        assert "1024 bytes" in reported[0]

    async def test_a_catalog_published_by_another_build_is_not_read(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A different version or configuration is a different fingerprint, so a miss.

        This is what keeps a rolling deployment from feeding one build's
        catalog to another.

        Ref: stdapi/models/_shared_cache.py:fingerprint
        """
        del dynamodb_table
        self._enable(monkeypatch)
        publisher = catalog("vendor.one")
        await initialize_bedrock_models()
        assert publisher.calls == 1

        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_regions", ["eu-central-1", "us-east-1"]
        )
        monkeypatch.setitem(stdapi.models._CACHE, "update_next", None)  # noqa: SLF001
        monkeypatch.setitem(stdapi.models._CACHE, "updated_at", None)  # noqa: SLF001
        reader = catalog("vendor.one")

        await initialize_bedrock_models()

        assert reader.calls == 1

    async def test_an_expired_published_catalog_is_not_read(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manifest's own age is the staleness rule, not the table's expiry.

        DynamoDB deletes an expired item eventually rather than on time, so a
        reader that trusted the time-to-live could read an arbitrarily old
        catalog.

        Ref: stdapi/models/_shared_cache.py:read_catalog
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html
        """
        del dynamodb_table
        self._enable(monkeypatch)
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )
        stale = int(datetime.now(UTC).timestamp()) - SETTINGS.model_cache_seconds - 1

        await _shared_cache.publish_catalog(payload, stale, None)

        assert await _shared_cache.read_catalog(None) is None

    async def test_a_published_list_outlives_the_age_readers_accept_it_until(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The table's expiry never deletes a manifest a reader would still use.

        A time-to-live shorter than ``model_cache_seconds`` drops the manifest
        mid-interval and sends the whole fleet back to sweeping for itself for
        the rest of it, silently: a missing manifest is an ordinary miss and
        says nothing to anyone.

        Ref: stdapi/models/_shared_cache.py:publish_catalog
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html
        """
        del dynamodb_table
        self._enable(monkeypatch)
        # A lifetime past the other two floors, and no stale window at all.
        monkeypatch.setattr(SETTINGS, "model_cache_seconds", 7200)
        monkeypatch.setattr(SETTINGS, "model_cache_max_stale_seconds", 0)
        created_at = int(datetime.now(UTC).timestamp())
        payload = stdapi.models._catalog_payload(  # noqa: SLF001
            {"vendor.one": make_model_details("vendor.one")}, {}
        )

        await _shared_cache.publish_catalog(payload, created_at, None)

        manifest = await get_item(
            item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint()), "manifest"
        )
        assert manifest is not None
        expires_at = manifest[EXPIRES_AT_ATTRIBUTE]
        assert isinstance(expires_at, int)
        assert expires_at >= created_at + 7200

    async def test_one_server_wins_the_right_to_sweep(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent servers produce exactly one sweeper, and it can hand the lease back.

        Ref: stdapi/models/_shared_cache.py:acquire_lease
        """
        del dynamodb_table
        self._enable(monkeypatch)

        outcomes = await gather(*(_shared_cache.acquire_lease(None) for _ in range(4)))

        assert outcomes.count(_shared_cache.Lease.HELD) == 1
        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.PEER

        await _shared_cache.release_lease(None)

        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.HELD

    async def test_a_server_that_loses_the_lease_serves_what_it_has(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A peer already sweeping means this server sweeps neither.

        Ref: stdapi/models/__init__.py:_collect_catalog
        """
        del dynamodb_table
        # Primed with sharing off, so the table holds no list this server could
        # read: the peer's lease is what has to stop the sweep here, and a
        # readable list would settle it before the lease is ever consulted.
        sweep = catalog("vendor.one")
        await initialize_bedrock_models()
        assert sweep.calls == 1
        self._enable(monkeypatch)
        _age_catalog(60)
        # A peer takes the lease, and publishes nothing yet.
        monkeypatch.setattr(server, "SERVER_NAME", "another-server")
        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.HELD
        monkeypatch.setattr(server, "SERVER_NAME", "this-server")

        await stdapi.models._refresh_bedrock_models(None)  # noqa: SLF001

        assert sweep.calls == 1
        assert "vendor.one" in stdapi.models._MODELS  # noqa: SLF001

    async def test_the_staleness_ceiling_outranks_the_peers_lease(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Past the ceiling a server sweeps for itself rather than wait on a peer.

        Waiting is only allowed while there is a list this server may still
        answer with. Past ``model_cache_max_stale_seconds`` there is not, and
        deferring would serve for another lease period exactly what the ceiling
        exists to stop -- on every server of a fleet but the one holding the
        lease, which is where the promise would quietly stop holding.

        Ref: stdapi/models/__init__.py:_collect_catalog
             stdapi/config.py:_Settings.model_cache_max_stale_seconds
        """
        del dynamodb_table
        # Primed with sharing off: nothing publishable in the table, so the
        # lease is what this server has to get past.
        first = catalog("vendor.one")
        await initialize_bedrock_models()
        assert first.calls == 1
        self._enable(monkeypatch)
        _age_catalog(60, since_success=SETTINGS.model_cache_max_stale_seconds + 60)
        monkeypatch.setattr(server, "SERVER_NAME", "another-server")
        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.HELD
        monkeypatch.setattr(server, "SERVER_NAME", "this-server")
        sweep = catalog("vendor.two")

        await initialize_bedrock_models()

        assert sweep.calls == 1
        assert "vendor.one" not in stdapi.models._MODELS  # noqa: SLF001

    async def test_a_failure_that_comes_back_is_reported_again(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suppression lasts while the fault lasts, not for the life of the process.

        The troubleshooting page promises every failure is reported at
        ``WARNING``. Reported-once-forever would keep that promise only for the
        first occurrence: a permission revoked, restored and revoked again would
        be silent the second time while every refresh fell back to a local
        sweep.

        Ref: docs/operations_troubleshooting.md
             stdapi/models/_shared_cache.py:_warn
        """
        del dynamodb_table
        self._enable(monkeypatch)
        reported: list[str] = []
        monkeypatch.setattr(_shared_cache, "_REPORTED", set())
        monkeypatch.setattr(
            _shared_cache,
            "log_error_details",
            lambda detail, level: reported.append(detail),  # noqa: ARG005
        )

        _shared_cache._warn(None, "the table is not readable")  # noqa: SLF001
        _shared_cache._warn(None, "the table is not readable")  # noqa: SLF001
        assert reported == ["the table is not readable"], "repeats buried the log"

        # Forgetting is wired to the table answering, not left to a caller to
        # remember: this is the call every refresh round makes.
        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.HELD
        _shared_cache._warn(None, "the table is not readable")  # noqa: SLF001

        assert reported == ["the table is not readable"] * 2

    async def test_the_lease_records_the_name_the_server_logs_under(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorded holder is read late, so it matches what the logs show.

        On ECS the server renames itself during startup, after this module has
        been imported. A holder captured at import time would name a server
        that appears in no log line and matches no task, which is the one thing
        an operator has to correlate when a lease is blocking refreshes.

        Ref: stdapi/aws.py:_set_ecs_metadata
             stdapi/models/_shared_cache.py:acquire_lease
        """
        del dynamodb_table
        self._enable(monkeypatch)
        monkeypatch.setattr(server, "SERVER_NAME", "task-abc-container-1")

        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.HELD

        record = await get_item(
            item_key(_shared_cache.NAMESPACE, _shared_cache.fingerprint()),
            _shared_cache._LEASE,  # noqa: SLF001
        )
        assert record is not None
        assert record["lease_holder"] == "task-abc-container-1"

    async def test_a_cold_server_sweeps_even_without_the_lease(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A server with nothing to serve never waits on a peer.

        Ref: stdapi/models/__init__.py:_collect_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        monkeypatch.setattr(server, "SERVER_NAME", "another-server")
        assert await _shared_cache.acquire_lease(None) is _shared_cache.Lease.HELD
        monkeypatch.setattr(server, "SERVER_NAME", "this-server")
        sweep = catalog("vendor.one")

        await initialize_bedrock_models()

        assert sweep.calls == 1

    async def test_a_table_failure_falls_back_to_discovering(
        self,
        catalog: Callable[..., _Sweep],
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """An unusable table costs a warning and a sweep, never a failed request.

        Ref: stdapi/models/_shared_cache.py:read_catalog
             stdapi/aws_dynamodb.py:TableUnavailableError
        """
        self._enable(monkeypatch)
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "no-such-table")
        sweep = catalog("vendor.one")

        await initialize_bedrock_models()

        assert sweep.calls == 1
        assert "vendor.one" in stdapi.models._MODELS  # noqa: SLF001
        assert "aws_dynamodb_table" in str(request_log["error_detail"])

    async def test_a_warm_server_sweeps_when_the_table_stops_answering(
        self,
        catalog: Callable[..., _Sweep],
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A table that goes away is not a peer: nobody else is going to publish.

        The two look the same from the lease write -- neither lets this server
        publish -- but only a peer produces a catalog. Reading a failure as a
        peer leaves a warm server refreshing never again and serving a list
        past the staleness ceiling it promises.

        Ref: stdapi/models/_shared_cache.py:acquire_lease
             stdapi/models/__init__.py:_collect_catalog
        """
        del dynamodb_table
        self._enable(monkeypatch)
        sweep = catalog("vendor.one")
        await initialize_bedrock_models()
        assert sweep.calls == 1
        _age_catalog(60)
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "no-such-table")

        await stdapi.models._refresh_bedrock_models(None)  # noqa: SLF001

        assert sweep.calls == 2
        assert "vendor.one" in stdapi.models._MODELS  # noqa: SLF001
