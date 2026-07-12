"""Unit tests for Polly multi-region voice initialization and failover routing."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ClientError

import stdapi.aws
from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.models import EXTRA_MODELS
from stdapi.models.audio import amazon_polly
from stdapi.models.audio.amazon_polly import (
    _engine_voice_regions,
    initialize_polly_models,
)
from stdapi.monitoring import EventLog

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_polly.literals import EngineType, VoiceIdType


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


#: Module dictionaries mutated by initialize_polly_models.
_STATE_DICTS = (
    "_VOICES_DESCRIPTIONS",
    "_VOICES_BY_GENDERS",
    "_VOICES_BY_LANGUAGE",
    "_VOICES_BY_ENGINE",
    "_VOICES_BY_NAME_LOWER",
    "_VOICES_BY_ENGINE_REGION",
    "EXTRA_MODELS",
    "EXTRA_MODELS_INPUT_MODALITY",
    "EXTRA_MODELS_OUTPUT_MODALITY",
)


def _copy(value: object) -> object:
    """Deep-copy the set/dict-of-set values mutated in place by init."""
    if isinstance(value, set):
        return set(value)
    if isinstance(value, dict):
        return {key: _copy(inner) for key, inner in value.items()}
    return value


@pytest.fixture(autouse=True)
def _isolated_polly_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Pin candidate regions, purge session Polly models, restore state after.

    The session app may already have registered real ``amazon.polly-*``
    entries; they are removed so each test observes a from-scratch init.
    """
    monkeypatch.setattr(SETTINGS, "aws_polly_region", None)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
    saved = {name: _copy(getattr(amazon_polly, name)) for name in _STATE_DICTS}
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
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[tuple[str, str], set[str] | Exception],
) -> None:
    """Fake per-(engine, region) voice retrieval; unlisted pairs are empty."""

    async def _fake(engine: EngineType, region: RegionName) -> set[VoiceIdType]:
        outcome = outcomes.get((engine, region), set())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

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


def _engines_warning(start_event: EventLog) -> Mapping[str, object]:
    (warning,) = start_event["server_warnings"]
    assert isinstance(warning, dict)
    failed = warning["unavailable_polly_engines"]
    assert isinstance(failed, dict)
    return failed


class TestInitializePollyModels:
    """initialize_polly_models: per-region engine discovery and registration."""

    async def test_engine_regions_follow_candidate_priority_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine found in both regions lists them in candidate order."""
        _patch_voices(
            monkeypatch,
            {
                ("standard", "us-east-1"): {"Joanna"},
                ("standard", "eu-west-1"): {"Joanna", "Lea"},
            },
        )
        await initialize_polly_models()

        model = EXTRA_MODELS["amazon.polly-standard"]
        assert model.regions == ["us-east-1", "eu-west-1"]
        assert amazon_polly._VOICES_BY_ENGINE["standard"] == {  # noqa: SLF001
            "Joanna",
            "Lea",
        }

    async def test_engine_available_in_one_region_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine with voices in a single region is scoped to it."""
        _patch_voices(monkeypatch, {("generative", "eu-west-1"): {"Lea"}})
        await initialize_polly_models()

        assert EXTRA_MODELS["amazon.polly-generative"].regions == ["eu-west-1"]
        assert "amazon.polly-standard" not in EXTRA_MODELS

    async def test_engine_with_no_voices_anywhere_is_not_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty voice lists everywhere (unsupported engine): no model, no warning."""
        _patch_voices(monkeypatch, {})
        start_event = _start_event()

        await initialize_polly_models(start_event)

        assert not any(m.startswith("amazon.polly-") for m in EXTRA_MODELS)
        assert "server_warnings" not in start_event

    async def test_failed_region_is_warned_and_the_other_still_serves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One region failing keeps the engine served from the other region."""
        _patch_voices(
            monkeypatch,
            {("neural", "us-east-1"): _aws_error(), ("neural", "eu-west-1"): {"Lea"}},
        )
        start_event = _start_event()

        await initialize_polly_models(start_event)

        assert EXTRA_MODELS["amazon.polly-neural"].regions == ["eu-west-1"]
        assert list(_engines_warning(start_event)) == ["neural@us-east-1"]

    async def test_all_pairs_failing_still_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every (engine, region) failing yields warnings, never a crash."""
        engines = ("standard", "neural", "long-form", "generative")
        _patch_voices(
            monkeypatch,
            {
                (engine, region): _aws_error()
                for engine in engines
                for region in ("us-east-1", "eu-west-1")
            },
        )
        start_event = _start_event()

        await initialize_polly_models(start_event)

        assert not any(m.startswith("amazon.polly-") for m in EXTRA_MODELS)
        assert len(_engines_warning(start_event)) == 8

    async def test_non_aws_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Programming errors are never swallowed as a region failure."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): ValueError("bug")})

        with pytest.raises(ValueError, match="bug"):
            await initialize_polly_models(_start_event())


class TestEngineVoiceRegions:
    """_engine_voice_regions: synthesis routes to regions offering the voice."""

    @pytest.fixture(autouse=True)
    def _seed_engine_regions(self) -> None:
        """Replace any session-discovered voice data with a fixed matrix."""
        amazon_polly._VOICES_BY_ENGINE_REGION.clear()  # noqa: SLF001
        amazon_polly._VOICES_BY_ENGINE_REGION["neural"] = {  # noqa: SLF001
            "us-east-1": {"Joanna"},
            "eu-west-1": {"Joanna", "Lea"},
        }

    def test_voice_present_in_both_regions_keeps_priority_order(self) -> None:
        """A voice available everywhere routes to all regions, in order."""
        assert _engine_voice_regions("neural", "Joanna") == ["us-east-1", "eu-west-1"]

    def test_voice_present_in_one_region_routes_there_only(self) -> None:
        """A region-specific voice routes only to its region."""
        assert _engine_voice_regions("neural", "Lea") == ["eu-west-1"]

    def test_unknown_voice_falls_back_to_engine_regions(self) -> None:
        """An undiscovered voice name tries every region offering the engine."""
        assert _engine_voice_regions("neural", "Ghost") == ["us-east-1", "eu-west-1"]

    def test_unknown_engine_falls_back_to_all_candidate_regions(self) -> None:
        """An engine with no discovered region uses the full candidate list."""
        assert _engine_voice_regions("standard", "Joanna") == ["us-east-1", "eu-west-1"]


class _StubComprehendClient:
    """Stub Comprehend client with a fixed per-region behavior."""

    def __init__(self, outcome: dict[str, object] | Exception) -> None:
        self._outcome = outcome

    async def detect_dominant_language(self, Text: str) -> dict[str, object]:  # noqa: N803
        """Return the fixed payload or raise the configured error."""
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class TestDetectLanguageFailover:
    """_detect_language: Comprehend fails over across candidate regions."""

    async def test_failover_serves_and_records_the_second_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first region falls over; usage records the second."""
        monkeypatch.setattr(SETTINGS, "default_tts_language", None)
        monkeypatch.setattr(SETTINGS, "aws_comprehend_region", None)
        clients: dict[str, _StubComprehendClient] = {
            "us-east-1": _StubComprehendClient(
                ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "x"}},
                    "DetectDominantLanguage",
                )
            ),
            "eu-west-1": _StubComprehendClient(
                {"Languages": [{"LanguageCode": "fr", "Score": 0.99}]}
            ),
        }
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, region=None: clients[region]
        )
        monkeypatch.setitem(
            amazon_polly._VOICES_BY_LANGUAGE,  # noqa: SLF001
            "fr-FR",
            {"Lea"},
        )
        usage_token = usage.init_usage()
        try:
            language = await amazon_polly._detect_language("bonjour")  # noqa: SLF001
            records = usage.USAGE.get()
        finally:
            usage.USAGE.reset(usage_token)

        assert language == "fr-FR"
        (key,) = records
        assert key.region == "eu-west-1"
