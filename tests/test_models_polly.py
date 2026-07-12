"""Unit tests for Polly voice initialization fault isolation."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ClientError

from stdapi.models import EXTRA_MODELS
from stdapi.models.audio import amazon_polly
from stdapi.models.audio.amazon_polly import initialize_polly_models
from stdapi.monitoring import EventLog

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_aiobotocore_polly.literals import EngineType

#: Voice dictionaries cleared/rebuilt by initialize_polly_models.
_VOICE_DICTS = (
    "_VOICES_DESCRIPTIONS",
    "_VOICES_BY_GENDERS",
    "_VOICES_BY_LANGUAGE",
    "_VOICES_BY_ENGINE",
    "_VOICES_BY_NAME_LOWER",
)

#: Model dictionaries in stdapi.models that Polly initialization mutates.
_MODEL_DICTS = (
    "EXTRA_MODELS",
    "EXTRA_MODELS_INPUT_MODALITY",
    "EXTRA_MODELS_OUTPUT_MODALITY",
)


class _StubPollyMeta:
    """Minimal client meta carrying only a region name."""

    region_name = "eu-west-3"


class _StubPollyClient:
    """Stub Polly client exposing only .meta.region_name."""

    meta = _StubPollyMeta()


def _snapshot(source: dict[str, object]) -> dict[str, object]:
    """Copy a module dict, deep-copying set values (mutated in place by init)."""
    return {
        key: set(value) if isinstance(value, set) else value
        for key, value in source.items()
    }


@pytest.fixture(autouse=True)
def _isolated_polly_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Stub the Polly client, purge session Polly models, restore state after.

    The session app may already have registered real ``amazon.polly-*``
    entries; they are removed so each test observes a from-scratch init.
    """
    monkeypatch.setattr(amazon_polly, "get_client", lambda _service: _StubPollyClient())
    saved = {
        name: _snapshot(getattr(amazon_polly, name))
        for name in _VOICE_DICTS + _MODEL_DICTS
    }
    for model_id in [m for m in EXTRA_MODELS if m.startswith("amazon.polly-")]:
        del EXTRA_MODELS[model_id]
        for name in ("EXTRA_MODELS_INPUT_MODALITY", "EXTRA_MODELS_OUTPUT_MODALITY"):
            for model_ids in getattr(amazon_polly, name).values():
                model_ids.discard(model_id)
    yield
    for name, content in saved.items():
        target = getattr(amazon_polly, name)
        target.clear()
        target.update(content)


def _patch_voices(
    monkeypatch: pytest.MonkeyPatch, failing_engines: dict[str, Exception]
) -> None:
    """Fake per-engine voice retrieval, raising for the given engines."""

    async def _fake(engine: EngineType) -> None:
        # Mimic the real function: register the dict entry before failing so
        # partial pagination cleanup is exercised.
        amazon_polly._VOICES_BY_ENGINE[engine] = {"Lea"}  # noqa: SLF001
        if (error := failing_engines.get(engine)) is not None:
            raise error

    monkeypatch.setattr(amazon_polly, "_get_voices_per_engine", _fake)


def _start_event() -> EventLog:
    return EventLog(
        type="start",
        level="info",
        date=datetime.now(UTC),
        server_id="test",
        server_version="0.0.0",
    )


def _aws_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "DescribeVoices",
    )


class TestInitializePollyModelsFaultIsolation:
    """initialize_polly_models: a failing engine must not fail startup."""

    async def test_failed_engine_is_disabled_and_warned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing engine is dropped with a warning; the others register."""
        _patch_voices(monkeypatch, {"neural": _aws_error()})
        start_event = _start_event()

        await initialize_polly_models(start_event)

        assert "amazon.polly-neural" not in EXTRA_MODELS
        assert "amazon.polly-standard" in EXTRA_MODELS
        (warning,) = start_event["server_warnings"]
        assert isinstance(warning, dict)
        engines = warning["unavailable_polly_engines"]
        assert isinstance(engines, dict)
        assert list(engines) == ["neural"]
        # The partially retrieved voice set was dropped, not half-kept.
        assert "neural" not in amazon_polly._VOICES_BY_ENGINE  # noqa: SLF001

    async def test_all_engines_failing_still_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with every engine failing, initialization completes with warnings."""
        engines = ("standard", "neural", "long-form", "generative")
        _patch_voices(monkeypatch, dict.fromkeys(engines, _aws_error()))
        start_event = _start_event()

        await initialize_polly_models(start_event)

        assert not any(
            model_id.startswith("amazon.polly-") for model_id in EXTRA_MODELS
        )
        (warning,) = start_event["server_warnings"]
        assert isinstance(warning, dict)
        failed = warning["unavailable_polly_engines"]
        assert isinstance(failed, dict)
        assert sorted(failed) == sorted(engines)

    async def test_non_aws_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Programming errors are never swallowed as an engine failure."""
        _patch_voices(monkeypatch, {"neural": ValueError("bug")})

        with pytest.raises(ValueError, match="bug"):
            await initialize_polly_models(_start_event())

    async def test_no_start_event_stays_silent_but_tolerant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a start event the failure is still tolerated (no crash)."""
        _patch_voices(monkeypatch, {"neural": _aws_error()})

        await initialize_polly_models()

        assert "amazon.polly-standard" in EXTRA_MODELS
