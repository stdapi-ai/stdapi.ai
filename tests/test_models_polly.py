"""Unit tests for Polly multi-region voice initialization and failover routing."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ClientError

import stdapi.aws
from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.models import EXTRA_MODELS
from stdapi.models.audio import amazon_polly, get_audio_model
from stdapi.models.audio.amazon_polly import (
    _engine_voice_regions,
    _PollyExtraParams,
    _select_voice,
    initialize_polly_models,
)
from stdapi.monitoring import REQUEST_LOG, EventLog

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


def _patch_describe_voices(
    monkeypatch: pytest.MonkeyPatch,
    scripts: Mapping[tuple[str, str], list[dict[str, object] | Exception]],
) -> None:
    """Fake per-region Polly clients, scripting describe_voices page sequences.

    Exercises the real ``_get_voices_per_engine`` pagination loop. Pairs not
    listed in *scripts* return a single empty page.
    """

    class _FakePollyClient:
        """Fake Polly client returning scripted describe_voices pages."""

        def __init__(self, region: RegionName) -> None:
            self._region = region
            self._calls: dict[str, int] = {}

        async def describe_voices(self, **params: object) -> dict[str, object]:
            """Return the next scripted page for the engine, or raise it."""
            engine = params["Engine"]
            assert isinstance(engine, str)
            call_index = self._calls.get(engine, 0)
            self._calls[engine] = call_index + 1
            page = scripts.get((engine, self._region), [{"Voices": []}])[call_index]
            if isinstance(page, Exception):
                raise page
            return page

    def _fake_get_client(_service: str, region: RegionName) -> _FakePollyClient:
        return _FakePollyClient(region)

    monkeypatch.setattr(amazon_polly, "get_client", _fake_get_client)


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

    async def test_page_failure_discards_earlier_pages_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second-page error leaves no metadata from the first page behind."""
        partial_voice = {
            "Id": "Partial",
            "Gender": "Female",
            "LanguageName": "English",
            "LanguageCode": "en-US",
        }
        full_voice = {
            "Id": "Full",
            "Gender": "Male",
            "LanguageName": "French",
            "LanguageCode": "fr-FR",
        }
        _patch_describe_voices(
            monkeypatch,
            {
                ("standard", "us-east-1"): [
                    {"Voices": [partial_voice], "NextToken": "page2"},
                    _aws_error(),
                ],
                ("neural", "us-east-1"): [{"Voices": [full_voice]}],
            },
        )
        start_event = _start_event()

        await initialize_polly_models(start_event)

        assert "Partial" not in amazon_polly._VOICES_DESCRIPTIONS  # noqa: SLF001
        assert "Partial" not in amazon_polly._VOICES_BY_GENDERS.get(  # noqa: SLF001
            "Female", set()
        )
        assert "Partial" not in amazon_polly._VOICES_BY_LANGUAGE.get(  # noqa: SLF001
            "en-US", set()
        )
        assert "partial" not in amazon_polly._VOICES_BY_NAME_LOWER  # noqa: SLF001
        assert "Partial" not in amazon_polly._VOICES_BY_ENGINE.get(  # noqa: SLF001
            "standard", set()
        )
        assert "amazon.polly-standard" not in EXTRA_MODELS
        assert list(_engines_warning(start_event)) == ["standard@us-east-1"]

        assert amazon_polly._VOICES_BY_ENGINE["neural"] == {"Full"}  # noqa: SLF001
        assert EXTRA_MODELS["amazon.polly-neural"].regions == ["us-east-1"]

    async def test_multi_page_listing_merges_voices_from_every_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NextToken-paginated listing accumulates voices from all pages."""
        page1_voice = {
            "Id": "Joanna",
            "Gender": "Female",
            "LanguageName": "English",
            "LanguageCode": "en-US",
        }
        page2_voice = {
            "Id": "Matthew",
            "Gender": "Male",
            "LanguageName": "English",
            "LanguageCode": "en-US",
        }
        _patch_describe_voices(
            monkeypatch,
            {
                ("standard", "us-east-1"): [
                    {"Voices": [page1_voice], "NextToken": "page2"},
                    {"Voices": [page2_voice]},
                ]
            },
        )

        await initialize_polly_models()

        assert amazon_polly._VOICES_BY_ENGINE["standard"] == {  # noqa: SLF001
            "Joanna",
            "Matthew",
        }
        assert amazon_polly._VOICES_BY_ENGINE_REGION["standard"]["us-east-1"] == {  # noqa: SLF001
            "Joanna",
            "Matthew",
        }
        assert amazon_polly._VOICES_DESCRIPTIONS["Joanna"] == "Female, English"  # noqa: SLF001
        assert amazon_polly._VOICES_DESCRIPTIONS["Matthew"] == "Male, English"  # noqa: SLF001
        assert amazon_polly._VOICES_BY_GENDERS["Female"] == {"Joanna"}  # noqa: SLF001
        assert amazon_polly._VOICES_BY_GENDERS["Male"] == {"Matthew"}  # noqa: SLF001
        assert amazon_polly._VOICES_BY_LANGUAGE["en-US"] == {  # noqa: SLF001
            "Joanna",
            "Matthew",
        }
        assert amazon_polly._VOICES_BY_NAME_LOWER["joanna"] == "Joanna"  # noqa: SLF001
        assert amazon_polly._VOICES_BY_NAME_LOWER["matthew"] == "Matthew"  # noqa: SLF001


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


class TestSelectVoiceDeterminism:
    """_select_voice: the detected language always outranks the en-US fallback."""

    @pytest.fixture(autouse=True)
    def _seed_voice_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Populate matching female neural voices for both fr-FR and en-US."""
        female_voices = amazon_polly._VOICES_BY_GENDERS  # noqa: SLF001
        by_language = amazon_polly._VOICES_BY_LANGUAGE  # noqa: SLF001
        by_engine = amazon_polly._VOICES_BY_ENGINE  # noqa: SLF001
        monkeypatch.setitem(female_voices, "Female", {"Lea", "Joanna"})
        monkeypatch.setitem(by_language, "fr-FR", {"Lea"})
        monkeypatch.setitem(by_language, "en-US", {"Joanna"})
        monkeypatch.setitem(by_engine, "neural", {"Lea", "Joanna"})

    async def test_detected_language_wins_over_en_us_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detected non-English language is never overridden by en-US, every run."""

        async def _detect(_text: str) -> str:
            return "fr-FR"

        monkeypatch.setattr(amazon_polly, "_detect_language", _detect)

        for _ in range(20):
            assert await _select_voice("Bonjour", "alloy", "neural") == ("Lea", "fr-FR")

    async def test_english_detection_still_selects_en_us(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detected en-US language (matching the fallback) still selects it."""

        async def _detect(_text: str) -> str:
            return "en-US"

        monkeypatch.setattr(amazon_polly, "_detect_language", _detect)

        assert await _select_voice("Hello", "alloy", "neural") == ("Joanna", "en-US")


class TestPollyExtraParamsLexiconNames:
    """_PollyExtraParams.LexiconNames: accepts and forwards the documented list form."""

    def test_list_value_is_accepted_and_dumps_as_a_list(self) -> None:
        """The documented ``["MyLexicon"]`` list form validates and round-trips."""
        extra = _PollyExtraParams(LexiconNames=["MyCustomLexicon"])
        assert extra.model_dump(exclude_none=True) == {
            "LexiconNames": ["MyCustomLexicon"]
        }


class TestPollyExtraParamsSpeechMarkTypes:
    """_PollyExtraParams.SpeechMarkTypes: forwarded to Polly verbatim, not value-constrained."""

    def test_documented_mark_types_are_accepted_and_dump_as_a_list(self) -> None:
        """The four Polly mark types validate and round-trip unchanged."""
        extra = _PollyExtraParams(
            SpeechMarkTypes=["sentence", "ssml", "viseme", "word"]
        )
        assert extra.model_dump(exclude_none=True) == {
            "SpeechMarkTypes": ["sentence", "ssml", "viseme", "word"]
        }

    def test_unknown_mark_type_is_forwarded_verbatim(self) -> None:
        """A mark type Polly does not define is not rejected here.

        Polly's own MarksNotSupportedForFormatException already maps an
        unsupported value to a 400 error; duplicating that check here would
        only reject values a future Polly release might add.
        """
        extra = _PollyExtraParams(SpeechMarkTypes=["phoneme"])
        assert extra.SpeechMarkTypes == ["phoneme"]


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


class _FakeAudioStream:
    """Fake Polly audio stream body, yielding one chunk then EOF."""

    def __init__(self, data: bytes) -> None:
        self._data: bytes | None = data

    async def read(self, _size: int) -> bytes:
        """Return the chunk once, then empty bytes."""
        data, self._data = self._data, None
        return data or b""


class _StubPollyClient:
    """Stub Polly client with a fixed per-region synthesize_speech outcome."""

    def __init__(self, outcome: dict[str, object] | Exception) -> None:
        self._outcome = outcome
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    async def synthesize_speech(self, **kwargs: object) -> dict[str, object]:
        """Record the request and return the fixed payload or raise the error."""
        self.calls += 1
        self.requests.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class TestSynthesizeSpeechFailover:
    """AudioModel.tts: end-to-end synthesis fails over across candidate regions."""

    async def test_failover_serves_audio_and_records_the_second_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first region falls over; usage and the log name it."""
        _patch_voices(
            monkeypatch,
            {("neural", "us-east-1"): {"Joanna"}, ("neural", "eu-west-1"): {"Joanna"}},
        )
        await initialize_polly_models()
        # _patch_voices bypasses the real metadata merge; seed the lookup
        # _select_voice needs to route "Joanna" straight to that voice ID.
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        clients = {
            "us-east-1": _StubPollyClient(
                ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "x"}},
                    "SynthesizeSpeech",
                )
            ),
            "eu-west-1": _StubPollyClient(
                {
                    "AudioStream": _FakeAudioStream(b"audio-bytes"),
                    "RequestCharacters": "5",
                }
            ),
        }
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, region=None: clients[region]
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            response = await get_audio_model("amazon.polly-neural").tts(
                text="Hello", voice="Joanna", resp_format="mp3"
            )
            audio = b"".join([chunk async for chunk in response["audio_stream"]])
            records = usage.USAGE.get()
            log = REQUEST_LOG.get()
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        assert audio == b"audio-bytes"
        assert clients["us-east-1"].calls == 1
        assert clients["eu-west-1"].calls == 1
        (key,) = records
        assert key.region == "eu-west-1"
        assert log["level"] == "warning"
        assert any(
            "polly" in str(detail) and "us-east-1" in str(detail)
            for detail in log["error_detail"]
        )


class TestSynthesizeSpeechEncodedFormats:
    """AudioModel.tts: wav/flac/aac are transcoded from lossless PCM, not Vorbis."""

    async def test_wav_request_synthesizes_from_pcm_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wav request asks Polly for pcm, not ogg_vorbis, with no SampleRate."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        # 0.5s of 16-bit mono silence at Polly's default 16 kHz pcm rate.
        client = _StubPollyClient(
            {
                "AudioStream": _FakeAudioStream(b"\x00\x00" * 8000),
                "RequestCharacters": "5",
            }
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            response = await get_audio_model("amazon.polly-neural").tts(
                text="Hello", voice="Joanna", resp_format="wav"
            )
            audio = b"".join([chunk async for chunk in response["audio_stream"]])
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert "SampleRate" not in request
        # ffmpeg successfully decoded the raw pcm at the assumed 16 kHz rate.
        assert audio.startswith(b"RIFF")

    async def test_explicit_sample_rate_overrides_the_pcm_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller-provided SampleRate is still forwarded for the pcm source."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        client = _StubPollyClient(
            {
                "AudioStream": _FakeAudioStream(b"\x00\x00" * 4000),
                "RequestCharacters": "5",
            }
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            await get_audio_model("amazon.polly-neural").tts(
                text="Hello",
                voice="Joanna",
                resp_format="flac",
                extra_params={"SampleRate": 8000},
            )
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert request["SampleRate"] == "8000"

    async def test_sample_rate_above_pcm_cap_falls_back_to_vorbis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SampleRate Polly pcm cannot serve keeps the Ogg Vorbis source."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        client = _StubPollyClient(
            {
                "AudioStream": _FakeAudioStream(b"\x00\x00" * 4000),
                "RequestCharacters": "5",
            }
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            await get_audio_model("amazon.polly-neural").tts(
                text="Hello",
                voice="Joanna",
                resp_format="flac",
                extra_params={"SampleRate": 24000},
            )
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "ogg_vorbis"
        assert request["SampleRate"] == "24000"


class TestSynthesizeSpeechPcmSampleRate:
    """AudioModel.tts: default pcm output is forced to OpenAI's 24 kHz contract."""

    async def test_pcm_request_synthesizes_from_16khz_and_resamples_to_24khz(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default pcm request asks Polly for 16 kHz pcm, then resamples to 24 kHz."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        # 0.5s of 16-bit mono silence at Polly's default 16 kHz pcm rate.
        client = _StubPollyClient(
            {
                "AudioStream": _FakeAudioStream(b"\x00\x00" * 8000),
                "RequestCharacters": "5",
            }
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            response = await get_audio_model("amazon.polly-neural").tts(
                text="Hello", voice="Joanna", resp_format="pcm"
            )
            audio = b"".join([chunk async for chunk in response["audio_stream"]])
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert "SampleRate" not in request
        # 0.5s of silence: 16-bit mono at 24 kHz is exactly 24000 bytes.
        assert len(audio) == 24000

    async def test_pcm_explicit_sample_rate_bypasses_resampling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller-provided pcm SampleRate is forwarded and left untouched."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        raw_audio = b"\x01\x02" * 4000
        client = _StubPollyClient(
            {"AudioStream": _FakeAudioStream(raw_audio), "RequestCharacters": "5"}
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            response = await get_audio_model("amazon.polly-neural").tts(
                text="Hello",
                voice="Joanna",
                resp_format="pcm",
                extra_params={"SampleRate": 8000},
            )
            audio = b"".join([chunk async for chunk in response["audio_stream"]])
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert request["SampleRate"] == "8000"
        # No ffmpeg pass: the raw Polly bytes are streamed through verbatim.
        assert audio == raw_audio


class TestSynthesizeSpeechMarks:
    """AudioModel.tts: SpeechMarkTypes switches Polly to its JSON marks output."""

    async def test_speech_marks_request_json_output_and_content_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Marks force OutputFormat=json, bypass transcoding, and label the stream."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        marks = b'{"time":0,"type":"word","start":0,"end":5,"value":"Hello"}\n'
        client = _StubPollyClient(
            {"AudioStream": _FakeAudioStream(marks), "RequestCharacters": "5"}
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            response = await get_audio_model("amazon.polly-neural").tts(
                text="Hello",
                voice="Joanna",
                # "wav" would normally be transcoded from pcm by ffmpeg.
                resp_format="wav",
                extra_params={"SpeechMarkTypes": ["word"]},
            )
            payload = b"".join([chunk async for chunk in response["audio_stream"]])
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "json"
        assert request["SpeechMarkTypes"] == ["word"]
        assert payload == marks
        assert response["content_type"] == "application/x-json-stream"

    async def test_audio_request_has_no_content_type_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without marks, the response keeps deriving its type from the format."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        client = _StubPollyClient(
            {"AudioStream": _FakeAudioStream(b"audio"), "RequestCharacters": "5"}
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        log_token = REQUEST_LOG.set(_start_event())
        usage_token = usage.init_usage()
        try:
            response = await get_audio_model("amazon.polly-neural").tts(
                text="Hello", voice="Joanna", resp_format="mp3"
            )
            await response["audio_stream"].aclose()
        finally:
            usage.USAGE.reset(usage_token)
            REQUEST_LOG.reset(log_token)

        (request,) = client.requests
        assert request["OutputFormat"] == "mp3"
        assert response["content_type"] is None
