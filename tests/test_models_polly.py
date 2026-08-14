"""Unit tests for Polly multi-region voice initialization and failover routing.

Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
     https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
     stdapi/models/audio/amazon_polly.py
"""

from datetime import UTC, datetime
from shutil import which
from typing import TYPE_CHECKING

import pytest
from aws_sdk_polly.models import (
    AudioEvent,
    StartSpeechSynthesisStreamActionStreamCloseStreamEvent,
    StartSpeechSynthesisStreamEventStreamAudioEvent,
    StartSpeechSynthesisStreamEventStreamStreamClosedEvent,
    StreamClosedEvent,
    ValidationException,
    ValidationExceptionReason,
)
from botocore.exceptions import ClientError, ParamValidationError
from httpx import ASGITransport, AsyncClient

import stdapi.aws
import stdapi.aws_bidi
import stdapi.aws_s3
from stdapi import usage
from stdapi.api_errors import ApiError
from stdapi.cleanup import CLEANUPS
from stdapi.config import SETTINGS
from stdapi.models import EXTRA_MODELS
from stdapi.models.audio import amazon_polly, get_audio_model
from stdapi.models.audio.amazon_polly import (
    _engine_voice_regions,
    _PollyExtraParams,
    _select_voice,
    _stream_text_events,
    _synthesis_transport,
    initialize_polly_models,
)
from stdapi.monitoring import REQUEST_LOG, EventLog
from stdapi.pricing import Dimension
from tests._helpers import make_client_error, make_model_details
from tests.test_aws_bidi import FakeDuplexStream

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
    from typing import Any

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_polly.literals import EngineType, VoiceIdType


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture
def _requires_ffmpeg() -> None:
    """Skip a test that asserts on bytes only a real ffmpeg can produce."""
    if which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to re-encode Polly output")


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
    """initialize_polly_models: per-region engine discovery and registration.

    Engine and voice availability is a per-Region table (long-form exists only
    in us-east-1, generative in a subset), so the gateway must discover it with
    DescribeVoices rather than derive it.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
         https://docs.aws.amazon.com/polly/latest/dg/long-form-voices.html
         stdapi/models/audio/amazon_polly.py:initialize_polly_models
    """

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
        warning = _engines_warning(start_event)
        assert list(warning) == ["neural@us-east-1"]
        assert "AccessDeniedException" in str(warning["neural@us-east-1"]), (
            "the warning must carry the AWS error that made the region unusable"
        )

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
        warning = _engines_warning(start_event)
        assert sorted(warning) == sorted(
            f"{engine}@{region}"
            for engine in engines
            for region in ("us-east-1", "eu-west-1")
        )
        assert all("AccessDeniedException" in str(cause) for cause in warning.values())

    async def test_non_aws_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Programming errors are never swallowed as a region failure.

        Only ``ClientError`` / ``BotoCoreError`` are treated as an unavailable
        (engine, region) pair, so a non-AWS exception aborts startup before any
        model is registered and without being reported as a region warning.
        """
        _patch_voices(monkeypatch, {("neural", "us-east-1"): ValueError("bug")})
        start_event = _start_event()

        with pytest.raises(ValueError, match="bug") as excinfo:
            await initialize_polly_models(start_event)

        assert excinfo.value.args == ("bug",)
        assert "server_warnings" not in start_event
        assert not any(m.startswith("amazon.polly-") for m in EXTRA_MODELS)

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
    """_engine_voice_regions: synthesis routes to regions offering the voice.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
         stdapi/models/audio/amazon_polly.py:_engine_voice_regions
    """

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
    """_select_voice: the detected language always outranks the en-US fallback.

    OpenAI voice names have no Polly equivalent: they are resolved to a Polly
    voice of the same gender, in the detected language when one is available.

    Ref: https://developers.openai.com/api/docs/guides/text-to-speech#voice-options
         stdapi/models/audio/amazon_polly.py:_select_voice
    """

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
        """A detected non-English language is never overridden by the en-US fallback.

        ``dict.fromkeys`` preserves the detected language's insertion order ahead of
        the ``en-US`` fallback, so the outcome does not depend on set iteration order.

        Ref: https://docs.python.org/3/library/stdtypes.html#dict
             stdapi/models/audio/amazon_polly.py:_select_voice
        """

        async def _detect(_text: str) -> str:
            return "fr-FR"

        monkeypatch.setattr(amazon_polly, "_detect_language", _detect)

        assert await _select_voice("Bonjour", "alloy", "neural") == ("Lea", "fr-FR")

    async def test_english_detection_still_selects_en_us(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detected en-US language (matching the fallback) still selects it.

        Ref: https://docs.python.org/3/library/stdtypes.html#dict
             stdapi/models/audio/amazon_polly.py:_select_voice
        """

        async def _detect(_text: str) -> str:
            return "en-US"

        monkeypatch.setattr(amazon_polly, "_detect_language", _detect)

        assert await _select_voice("Hello", "alloy", "neural") == ("Joanna", "en-US")


class TestPollyExtraParamsLexiconNames:
    """_PollyExtraParams.LexiconNames: accepts and forwards the documented list form.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/managing-lexicons.html
         stdapi/models/audio/amazon_polly.py:_PollyExtraParams
    """

    def test_list_value_is_accepted_and_dumps_as_a_list(self) -> None:
        """The documented ``["MyLexicon"]`` list form validates and round-trips.

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
        """
        extra = _PollyExtraParams(LexiconNames=["MyCustomLexicon"])
        assert extra.model_dump(exclude_none=True) == {
            "LexiconNames": ["MyCustomLexicon"]
        }


class TestPollyExtraParamsSpeechMarkTypes:
    """_PollyExtraParams.SpeechMarkTypes: forwarded to Polly verbatim, not value-constrained.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/speechmarks.html
         https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/models/audio/amazon_polly.py:_PollyExtraParams
    """

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
    """_detect_language: Comprehend fails over across candidate regions.

    Ref: https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html
         stdapi/models/audio/amazon_polly.py:_detect_language
         stdapi/aws.py:call_with_region_failover
    """

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


@pytest.fixture
def _request_context() -> Generator[None]:
    """Bind the request-log and usage contexts the synthesis path writes into.

    ``AudioModel.tts`` logs the selected voice and records Polly usage, both of
    which read context variables that only exist inside a real request.
    """
    log_token = REQUEST_LOG.set(_start_event())
    usage_token = usage.init_usage()
    try:
        yield
    finally:
        usage.USAGE.reset(usage_token)
        REQUEST_LOG.reset(log_token)


@pytest.fixture
async def stub_polly(
    monkeypatch: pytest.MonkeyPatch, _request_context: None
) -> Callable[[bytes], _StubPollyClient]:
    """Register a single neural ``Joanna`` voice and stub the Polly client.

    ``_patch_voices`` bypasses the real metadata merge, so the lowercase lookup
    ``_select_voice`` uses is seeded explicitly.

    Returns:
        Factory binding a stub client that answers ``synthesize_speech`` with
        the given payload bytes, and returning it for request assertions.
    """
    _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
    await initialize_polly_models()
    amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

    def _bind(payload: bytes) -> _StubPollyClient:
        client = _StubPollyClient(
            {"AudioStream": _FakeAudioStream(payload), "RequestCharacters": "5"}
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        return client

    return _bind


@pytest.mark.usefixtures("_request_context")
class TestSynthesizeSpeechFailover:
    """AudioModel.tts: end-to-end synthesis fails over across candidate regions.

    SynthesizeSpeech throttles at 8 TPS on the neural engine, so a throttled
    Region must be retried elsewhere rather than surfaced to the caller.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/limits.html
         stdapi/aws.py:call_with_region_failover
    """

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
        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello", voice="Joanna", resp_format="mp3"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])
        records = usage.USAGE.get()
        log = REQUEST_LOG.get()

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


@pytest.mark.usefixtures("_request_context")
class TestSynthesizeSpeechParamValidationError:
    """AudioModel.tts: a client-side botocore rejection surfaces as a caller 400.

    Mirrors the Transcribe/Translate twins: a request field botocore itself
    rejects before reaching Polly (e.g. a malformed SampleRate) must not surface
    as an unhandled 500, and the caller gets a generic message rather than the
    raw botocore report (AGENTS.md "Never leak internals").

    Ref: botocore/data/polly/2016-06-10/service-2.json
         stdapi/models/audio/amazon_polly.py:_handle_polly_error
    """

    async def test_param_validation_error_becomes_a_caller_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ParamValidationError is mapped to a 400 with a fixed, generic message."""
        _patch_voices(monkeypatch, {("neural", "us-east-1"): {"Joanna"}})
        await initialize_polly_models()
        amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001

        client = _StubPollyClient(
            ParamValidationError(report="Invalid type for parameter SampleRate")
        )
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-neural").tts(
                text="Hello", voice="Joanna", resp_format="mp3"
            )

        assert excinfo.value.status == 400
        assert str(excinfo.value) == "Invalid speech synthesis settings.", (
            "the raw botocore report must not reach the caller"
        )
        log = REQUEST_LOG.get()
        assert any("SampleRate" in str(detail) for detail in log["error_detail"]), (
            "the botocore validation report must still be logged server-side"
        )


class TestSynthesizeSpeechEncodedFormats:
    """AudioModel.tts: wav/flac/aac are transcoded from lossless PCM, not Vorbis.

    Polly's OutputFormat has no wav/flac/aac, so those OpenAI formats are
    re-encoded in process; pcm is the lossless source, but it accepts only
    8 kHz and 16 kHz, hence the Ogg Vorbis fallback above that cap.

    Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/media.py:encode_audio_stream
    """

    @pytest.mark.usefixtures("_requires_ffmpeg")
    async def test_wav_request_synthesizes_from_pcm_by_default(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """A wav request asks Polly for pcm, not ogg_vorbis, with no SampleRate."""
        # 0.5s of 16-bit mono silence at Polly's default 16 kHz pcm rate.
        client = stub_polly(b"\x00\x00" * 8000)

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello", voice="Joanna", resp_format="wav"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert "SampleRate" not in request
        # ffmpeg successfully decoded the raw pcm at the assumed 16 kHz rate.
        assert audio.startswith(b"RIFF")

    @pytest.mark.parametrize(
        ("sample_rate", "expected_output_format"),
        [(8000, "pcm"), (24000, "ogg_vorbis")],
        ids=["within-pcm-cap", "above-pcm-cap"],
    )
    async def test_explicit_sample_rate_selects_the_source_format(
        self,
        stub_polly: Callable[[bytes], _StubPollyClient],
        sample_rate: int,
        expected_output_format: str,
    ) -> None:
        """A caller SampleRate is forwarded; above Polly's pcm cap the source is Vorbis.

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
        """
        client = stub_polly(b"\x00\x00" * 4000)

        await get_audio_model("amazon.polly-neural").tts(
            text="Hello",
            voice="Joanna",
            resp_format="flac",
            extra_params={"SampleRate": sample_rate},
        )

        (request,) = client.requests
        assert request["OutputFormat"] == expected_output_format
        assert request["SampleRate"] == str(sample_rate)

    async def test_vorbis_fallback_omits_pcm_only_encode_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_polly: Callable[[bytes], _StubPollyClient],
    ) -> None:
        """Above the pcm cap, channels/sample_rate are not passed to the Vorbis source.

        ``_ffmpeg_args`` only applies ``channels``/``sample_rate`` when
        ``input_format`` is set (the raw-pcm path); passing them alongside an
        autodetected Ogg Vorbis source (``input_format=None``) is misleading,
        even though ``_ffmpeg_args`` silently ignores them.

        Ref: stdapi/media.py:_ffmpeg_args
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
        stub_polly(b"vorbis-bytes")
        captured: dict[str, object] = {}

        async def _fake_encode_audio_stream(
            _stream: object, _output_format: str, **kwargs: object
        ) -> AsyncGenerator[bytes]:
            captured.update(kwargs)
            yield b""

        monkeypatch.setattr(
            amazon_polly, "encode_audio_stream", _fake_encode_audio_stream
        )

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello",
            voice="Joanna",
            resp_format="flac",
            extra_params={"SampleRate": 24000},
        )
        async for _chunk in response["audio_stream"]:
            pass

        assert captured["input_format"] is None
        assert captured["channels"] is None
        assert captured["sample_rate"] is None


class TestSynthesizeSpeechPcmSampleRate:
    """AudioModel.tts: default pcm output is forced to OpenAI's 24 kHz contract.

    Ref: https://stdapi.ai/api_openai_audio_speech/
         https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/models/audio/amazon_polly.py:AudioModel.tts
    """

    @pytest.mark.usefixtures("_requires_ffmpeg")
    async def test_pcm_request_synthesizes_from_16khz_and_resamples_to_24khz(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """A default pcm request asks Polly for 16 kHz pcm, then resamples to 24 kHz."""
        # 0.5s of 16-bit mono silence at Polly's default 16 kHz pcm rate.
        client = stub_polly(b"\x00\x00" * 8000)

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello", voice="Joanna", resp_format="pcm"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert "SampleRate" not in request
        # 0.5s of silence: 16-bit mono at 24 kHz is exactly 24000 bytes.
        assert len(audio) == 24000

    async def test_pcm_explicit_sample_rate_bypasses_resampling(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """A caller-provided pcm SampleRate is forwarded and left untouched."""
        raw_audio = b"\x01\x02" * 4000
        client = stub_polly(raw_audio)

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello",
            voice="Joanna",
            resp_format="pcm",
            extra_params={"SampleRate": 8000},
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        (request,) = client.requests
        assert request["OutputFormat"] == "pcm"
        assert request["SampleRate"] == "8000"
        # No ffmpeg pass: the raw Polly bytes are streamed through verbatim.
        assert audio == raw_audio

    async def test_zero_sample_rate_is_forwarded_as_a_string(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """SampleRate=0 is converted to the string ``"0"``, not left as an int.

        ``extra.SampleRate`` is falsy for ``0``, so a truthy check on it would skip
        the ``str()`` conversion and leave the model-dumped ``int`` in the request;
        botocore's client-side type validation rejects a non-``str`` SampleRate with
        a ``ParamValidationError`` (500 unless caught).

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
        """
        client = stub_polly(b"\x00\x00" * 4000)

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello",
            voice="Joanna",
            resp_format="pcm",
            extra_params={"SampleRate": 0},
        )
        await response["audio_stream"].aclose()

        (request,) = client.requests
        assert request["SampleRate"] == "0"


class TestSynthesizeSpeechMarks:
    """AudioModel.tts: SpeechMarkTypes switches Polly to its JSON marks output.

    A json OutputFormat response is served as application/x-json-stream, so no
    audio is generated and nothing is transcoded.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/speechmarks.html
         https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/models/audio/amazon_polly.py:AudioModel.tts
    """

    async def test_speech_marks_request_json_output_and_content_type(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """Marks force OutputFormat=json, bypass transcoding, and label the stream."""
        marks = b'{"time":0,"type":"word","start":0,"end":5,"value":"Hello"}\n'
        client = stub_polly(marks)

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello",
            voice="Joanna",
            # "wav" would normally be transcoded from pcm by ffmpeg.
            resp_format="wav",
            extra_params={"SpeechMarkTypes": ["word"]},
        )
        payload = b"".join([chunk async for chunk in response["audio_stream"]])

        (request,) = client.requests
        assert request["OutputFormat"] == "json"
        assert request["SpeechMarkTypes"] == ["word"]
        assert payload == marks
        assert response["content_type"] == "application/x-json-stream"

    async def test_audio_request_has_no_content_type_override(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """Without marks, the response keeps deriving its type from the format."""
        client = stub_polly(b"audio")

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello", voice="Joanna", resp_format="mp3"
        )
        await response["audio_stream"].aclose()

        (request,) = client.requests
        assert request["OutputFormat"] == "mp3"
        assert response["content_type"] is None


class TestSynthesizeSpeechTextType:
    """AudioModel.tts: an SSML document is declared to Polly as ``TextType=ssml``.

    Polly reads markup aloud verbatim when the request says ``TextType=text``,
    so a regression in the ``<speak>`` detection produces a 200 with valid
    audio of the tags being spoken -- invisible to a response-shape assertion.
    Speed is applied by wrapping plain text in a prosody envelope, which is
    SSML too.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/ssml.html
         https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html
         stdapi/models/audio/amazon_polly.py:_prepare_text_for_speech
    """

    async def test_ssml_document_is_forwarded_verbatim_as_ssml(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """Input starting with ``<speak>`` is sent unchanged with TextType=ssml."""
        client = stub_polly(b"audio")
        document = '<speak>Hello <break time="1s"/> world</speak>'

        response = await get_audio_model("amazon.polly-neural").tts(
            text=document, voice="Joanna", resp_format="mp3"
        )
        await response["audio_stream"].aclose()

        (request,) = client.requests
        assert request["TextType"] == "ssml"
        assert request["Text"] == document

    async def test_plain_text_is_sent_as_text(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """Plain text at the default speed is sent unwrapped with TextType=text."""
        client = stub_polly(b"audio")

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello", voice="Joanna", resp_format="mp3"
        )
        await response["audio_stream"].aclose()

        (request,) = client.requests
        assert request["TextType"] == "text"
        assert request["Text"] == "Hello"

    async def test_speed_wraps_plain_text_in_an_ssml_prosody_envelope(
        self, stub_polly: Callable[[bytes], _StubPollyClient]
    ) -> None:
        """A non-default speed becomes a ``<prosody rate>`` document, in percent."""
        client = stub_polly(b"audio")

        response = await get_audio_model("amazon.polly-neural").tts(
            text="Hello", voice="Joanna", resp_format="mp3", speed=1.5
        )
        await response["audio_stream"].aclose()

        (request,) = client.requests
        assert request["TextType"] == "ssml"
        assert request["Text"] == '<speak><prosody rate="150%">Hello</prosody></speak>'


class TestLongInputSelection:
    """_synthesis_transport: which way each input is synthesized.

    Both single-call limits are real and independent: 3,000 billed characters
    and 6,000 total, SSML tags counting only towards the second. A selector
    keyed on ``len(input)`` alone would push a tag-heavy but short document
    through a slower path, and let a 3,001-character one fail against Polly
    instead. Above them the answer depends on what the deployment can do:
    incremental synthesis has no storage requirement but a bounded duration,
    a job needs a bucket, and where neither can run the only limit that exists
    is the single call's.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/limits.html
         https://docs.aws.amazon.com/polly/latest/dg/bidirectional-streaming-choosing.html
         stdapi/models/audio/amazon_polly.py:_synthesis_transport
    """

    def test_input_at_the_billed_limit_stays_synchronous(self) -> None:
        """3,000 billed characters is accepted by SynthesizeSpeech itself."""
        assert (
            _synthesis_transport(
                "a" * 3000, "text", streamable=True, job_available=True
            )
            == "call"
        )

    def test_input_above_the_billed_limit_needs_another_path(self) -> None:
        """One character more than the billed limit is rejected by SynthesizeSpeech."""
        assert (
            _synthesis_transport(
                "a" * 3001, "text", streamable=False, job_available=True
            )
            == "job"
        )

    def test_a_streamable_long_input_is_streamed_rather_than_scheduled(self) -> None:
        """Incremental synthesis is preferred to a job: audio starts immediately."""
        assert (
            _synthesis_transport(
                "a" * 3001, "text", streamable=True, job_available=True
            )
            == "stream"
        )

    def test_a_streamable_input_needs_no_storage(self) -> None:
        """A deployment with no bucket still serves a long streamable input."""
        assert (
            _synthesis_transport(
                "a" * 3001, "text", streamable=True, job_available=False
            )
            == "stream"
        )

    def test_ssml_tags_are_not_billed_characters(self) -> None:
        """A document billed under the limit stays synchronous despite its tags."""
        document = f"<speak>{'<break time="1s"/>' * 100}{'a' * 2900}</speak>"

        assert len(document) > 3000, "the raw document must exceed the billed limit"
        assert (
            _synthesis_transport(document, "ssml", streamable=False, job_available=True)
            == "call"
        )

    def test_ssml_above_the_total_limit_needs_a_task(self) -> None:
        """Tags alone can exceed the 6,000-character total limit."""
        document = f"<speak>hi{'<break time="1s"/>' * 340}</speak>"

        assert len(document) > 6000
        assert (
            _synthesis_transport(document, "ssml", streamable=False, job_available=True)
            == "job"
        )

    def test_beyond_one_stream_the_input_is_scheduled_instead(self) -> None:
        """A stream lasts ten minutes, so longer text goes back to a job."""
        assert (
            _synthesis_transport(
                "a" * 20_001, "text", streamable=True, job_available=True
            )
            == "job"
        )

    def test_at_the_stream_limit_the_input_is_still_streamed(self) -> None:
        """The boundary itself is served incrementally."""
        assert (
            _synthesis_transport(
                "a" * 20_000, "text", streamable=True, job_available=True
            )
            == "stream"
        )

    def test_input_above_the_task_billed_limit_is_rejected(self) -> None:
        """Beyond 100,000 billed characters, the error states the limit."""
        with pytest.raises(ApiError) as excinfo:
            _synthesis_transport(
                "a" * 100_001, "text", streamable=False, job_available=True
            )

        assert excinfo.value.status == 400
        assert "100,000" in str(excinfo.value)

    def test_ssml_above_the_task_total_limit_is_rejected(self) -> None:
        """Beyond 200,000 total characters, the error states that limit too."""
        document = f"<speak>hi{'<break time="1s"/>' * 11_112}</speak>"

        assert len(document) > 200_000
        with pytest.raises(ApiError) as excinfo:
            _synthesis_transport(document, "ssml", streamable=False, job_available=True)

        assert excinfo.value.status == 400
        assert "200,000" in str(excinfo.value)

    @pytest.mark.parametrize("length", [3001, 100_001], ids=["above_call", "above_job"])
    def test_without_a_job_only_the_single_call_limit_exists(self, length: int) -> None:
        """No job and no stream means no other limit: both bands name 3,000.

        A server that cannot run a job never accepts 100,000 characters, so
        quoting that limit would advertise a length it always rejects.
        """
        with pytest.raises(ApiError) as excinfo:
            _synthesis_transport(
                "a" * length, "text", streamable=False, job_available=False
            )

        message = str(excinfo.value)
        assert excinfo.value.status == 400
        assert "3,000" in message
        assert "100,000" not in message

    def test_a_streamable_input_beyond_one_stream_names_the_stream_limit(self) -> None:
        """With no job to fall back on, the rejection states what does fit."""
        with pytest.raises(ApiError) as excinfo:
            _synthesis_transport(
                "a" * 20_001, "text", streamable=True, job_available=False
            )

        message = str(excinfo.value)
        assert excinfo.value.status == 400
        assert "20,000" in message
        assert "3,000" not in message, "this input may be far longer than one call"


class TestStreamTextEvents:
    """_stream_text_events: how one input becomes the events a stream carries.

    Each event carries at most the same 3,000 billed / 6,000 total characters
    a single call takes, and an SSML document may not span events, so the
    speed envelope is rebuilt around every chunk. Polly reassembles the text
    itself, so a chunk boundary does not have to be a sentence boundary.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/bidirectional-streaming-lifecycle.html
         stdapi/models/audio/amazon_polly.py:_stream_text_events
    """

    def test_short_text_is_sent_as_one_plain_event(self) -> None:
        """Text within a single event's limit is not split."""
        (event,) = _stream_text_events("Hello there.", 1.0)

        assert event.text == "Hello there."
        assert event.text_type == "text"

    def test_long_text_is_split_into_events_within_the_event_limit(self) -> None:
        """Every event stays under the per-event limit, and nothing is lost."""
        text = "hello world. " * 1000  # 13,000 characters

        events = _stream_text_events(text, 1.0)

        assert len(events) > 1
        assert all(len(event.text) <= 3000 for event in events)
        assert "".join(event.text for event in events) == text

    def test_chunks_break_on_a_space_rather_than_inside_a_word(self) -> None:
        """A word split across events would be pronounced as two."""
        text = "antidisestablishmentarianism " * 200  # 5,800 characters

        events = _stream_text_events(text, 1.0)

        assert len(events) == 2
        assert all(
            word == "antidisestablishmentarianism"
            for event in events
            for word in event.text.split()
        ), "a chunk boundary must not cut a word in half"
        assert "".join(event.text for event in events) == text

    def test_text_without_a_break_point_is_split_at_the_limit(self) -> None:
        """An unbreakable run must still be sent, not rejected."""
        events = _stream_text_events("a" * 4000, 1.0)

        assert [len(event.text) for event in events] == [3000, 1000]

    def test_a_chunk_that_opens_with_markup_is_still_plain_text(self) -> None:
        """Only the caller's own document is SSML, never a chunk boundary's text.

        A chunk starting on a literal ``<speak>`` inside prose would otherwise
        be sent as a document and rejected as malformed markup.
        """
        text = "word " * 599 + "<speak>and the rest is prose"

        events = _stream_text_events(text, 1.0)

        assert len(events) > 1
        assert all(event.text_type == "text" for event in events)
        assert "".join(event.text for event in events) == text

    def test_a_speed_envelope_is_rebuilt_around_every_chunk(self) -> None:
        """Each event is a self-contained SSML document, as the stream requires."""
        events = _stream_text_events("hello world. " * 1000, 1.5)

        assert len(events) > 1
        for event in events:
            assert event.text_type == "ssml"
            assert event.text.startswith('<speak><prosody rate="150%">')
            assert event.text.endswith("</prosody></speak>")


#: Task identifier the stubbed asynchronous synthesis answers with.
_JOB_ID = "1a2b3c4d"


class _StubSynthesisJobClient:
    """Stub Polly and S3 client scripting one asynchronous synthesis task."""

    def __init__(
        self,
        statuses: list[str],
        *,
        audio: bytes = b"long-audio",
        reason: str = "",
        start_error: Exception | None = None,
    ) -> None:
        self._statuses = statuses
        self._audio = audio
        self._reason = reason
        self._start_error = start_error
        self.output_uri = ""
        self.requests: list[dict[str, Any]] = []
        self.clients: list[tuple[str, str | None]] = []
        self.polls = 0
        self.fetched: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []

    def _task(self, status: str) -> dict[str, Any]:
        """Build the task description Polly answers with in *status*."""
        task: dict[str, Any] = {
            "TaskId": _JOB_ID,
            "TaskStatus": status,
            "OutputUri": self.output_uri,
            "RequestCharacters": 3001,
        }
        if status == "failed":
            task["TaskStatusReason"] = self._reason
        return task

    async def start_speech_synthesis_task(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and answer with a freshly scheduled task."""
        if self._start_error is not None:
            raise self._start_error
        self.requests.append(kwargs)
        # Polly answers with the URI of the object in the job's own region.
        region = self.clients[-1][1] if self.clients else "us-east-1"
        self.output_uri = (
            f"https://s3.{region}.amazonaws.com/{kwargs['OutputS3BucketName']}"
            f"/{kwargs['OutputS3KeyPrefix']}.{_JOB_ID}.mp3"
        )
        return {"SynthesisTask": self._task("scheduled")}

    async def get_speech_synthesis_task(self, **_kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Answer with the next scripted status, repeating the last one."""
        self.polls += 1
        return {
            "SynthesisTask": self._task(
                self._statuses[min(self.polls, len(self._statuses)) - 1]
            )
        }

    async def get_object(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Serve the synthesized object as a readable body."""
        self.fetched.append((kwargs["Bucket"], kwargs["Key"]))
        return {"Body": _FakeAudioStream(self._audio)}

    async def delete_object(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Record the deletion the scheduled cleanup performs."""
        self.deleted.append((kwargs["Bucket"], kwargs["Key"]))


@pytest.fixture
def pending_cleanups() -> Generator[list[Awaitable[None]]]:
    """Bind the per-request cleanup list, closing anything left unawaited."""
    pending: list[Awaitable[None]] = []
    token = CLEANUPS.set(pending)
    try:
        yield pending
    finally:
        CLEANUPS.reset(token)
        for task in pending:
            task.close()  # type: ignore[attr-defined]


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the poll loop's clock with one only its own backoff advances.

    Returns:
        The backoff delays waited so far, in order.
    """
    delays: list[float] = []
    now = 0.0

    async def _sleep(delay: float) -> None:
        """Record the backoff and advance the clock by it, instantly."""
        nonlocal now
        now += delay
        delays.append(delay)

    monkeypatch.setattr(amazon_polly, "sleep", _sleep)
    monkeypatch.setattr(amazon_polly, "monotonic", lambda: now)
    return delays


@pytest.fixture
async def stub_synthesis_job(
    monkeypatch: pytest.MonkeyPatch, _request_context: None
) -> Callable[[_StubSynthesisJobClient], _StubSynthesisJobClient]:
    """Register a neural ``Joanna`` voice in both regions, one with a bucket.

    Only the primary region has a bucket, so it is the only candidate unless a
    test configures another one.

    Returns:
        Factory binding the given stub as every AWS client the asynchronous
        path uses, and returning it for request assertions.
    """
    _patch_voices(
        monkeypatch,
        {("neural", "us-east-1"): {"Joanna"}, ("neural", "eu-west-1"): {"Joanna"}},
    )
    await initialize_polly_models()
    amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "primary-bucket")
    monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
    monkeypatch.setattr(
        stdapi.aws_s3, "BUCKET_TO_REGION", {"primary-bucket": "us-east-1"}
    )

    def _bind(client: _StubSynthesisJobClient) -> _StubSynthesisJobClient:
        def _get_client(service: str, region: str | None = None) -> object:
            """Serve the stub, recording which region's client was asked for."""
            client.clients.append((service, region))
            return client

        for module in (amazon_polly, stdapi.aws, stdapi.aws_s3):
            monkeypatch.setattr(module, "get_client", _get_client)
        return client

    return _bind


@pytest.mark.usefixtures("pending_cleanups")
class TestSynthesisJobPath:
    """AudioModel.tts: long input is synthesized as a job and streamed back.

    The audio is produced into the region's own bucket, so the whole path --
    job, poll, download, deletion -- must stay on the region that accepted the
    job, and the object is deleted once the request ends. A job that outlives
    the request writes after that deletion, and is expired by the prefix's
    lifecycle rule instead.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/asynchronous.html
         stdapi/models/audio/amazon_polly.py:AudioModel.tts
    """

    async def test_long_input_is_synthesized_and_the_object_cleaned_up(
        self,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
        pending_cleanups: list[Awaitable[None]],
        fake_clock: list[float],
    ) -> None:
        """The job's audio is served, billed once, and deleted after the response."""
        client = stub_synthesis_job(
            _StubSynthesisJobClient(["scheduled", "inProgress", "completed"])
        )

        response = await get_audio_model("amazon.polly-neural").tts(
            text="a" * 3001, voice="Joanna", resp_format="mp3"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        (request,) = client.requests
        assert request["OutputS3BucketName"] == "primary-bucket"
        assert request["Text"] == "a" * 3001
        assert request["OutputFormat"] == "mp3"
        assert audio == b"long-audio"
        assert response["input_tokens"] == 3001
        assert client.polls == 3
        assert fake_clock == [0.5, 1.0], "the poll interval must back off"
        (key,) = usage.USAGE.get()
        assert key.region == "us-east-1"

        # The audio lands under the prefix operators expire with a lifecycle rule.
        object_key = f"{SETTINGS.aws_s3_tmp_prefix}speech.{_JOB_ID}.mp3"
        assert request["OutputS3KeyPrefix"] == f"{SETTINGS.aws_s3_tmp_prefix}speech"
        assert client.fetched == [("primary-bucket", object_key)]
        assert client.deleted == [], "the deletion must not block the response"
        (cleanup,) = pending_cleanups
        await cleanup
        assert client.deleted == [("primary-bucket", object_key)]

    async def test_the_job_never_leaves_the_region_that_accepted_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
        pending_cleanups: list[Awaitable[None]],
        fake_clock: list[float],
    ) -> None:
        """A job served by the second region polls, reads and deletes there.

        Polly writes the audio to a bucket co-located with the job, so a
        region without one cannot serve the request at all, and every call
        that follows the job belongs to the region that took it.
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)
        monkeypatch.setattr(
            SETTINGS, "aws_s3_regional_buckets", {"eu-west-1": "fallback-bucket"}
        )
        monkeypatch.setattr(
            stdapi.aws_s3, "BUCKET_TO_REGION", {"fallback-bucket": "eu-west-1"}
        )
        client = stub_synthesis_job(
            _StubSynthesisJobClient(["inProgress", "completed"])
        )

        response = await get_audio_model("amazon.polly-neural").tts(
            text="a" * 3001, voice="Joanna", resp_format="mp3"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])
        (cleanup,) = pending_cleanups
        await cleanup

        assert audio == b"long-audio"
        assert fake_clock == [0.5], "the unfinished poll must have waited once"
        (request,) = client.requests
        assert request["OutputS3BucketName"] == "fallback-bucket"
        assert client.clients, "the AWS clients used must have been recorded"
        assert {region for _, region in client.clients} == {"eu-west-1"}, (
            "job, polling, download and deletion all belong to the serving region"
        )
        assert {service for service, _ in client.clients} == {"polly", "s3"}
        object_key = f"{SETTINGS.aws_s3_tmp_prefix}speech.{_JOB_ID}.mp3"
        assert client.fetched == [("fallback-bucket", object_key)]
        assert client.deleted == [("fallback-bucket", object_key)]
        (key,) = usage.USAGE.get()
        assert key.region == "eu-west-1"

    async def test_polling_backs_off_up_to_its_cap_then_times_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
        pending_cleanups: list[Awaitable[None]],
        fake_clock: list[float],
    ) -> None:
        """A job that never completes ends the request, leaving no object behind.

        The interval doubles from 0.5s but never exceeds 5s, so a long job is
        still noticed promptly without polling Polly hundreds of times, and
        ``ai_response_timeout`` bounds the whole wait.
        """
        timeout = 10
        monkeypatch.setattr(SETTINGS, "ai_response_timeout", timeout)
        client = stub_synthesis_job(_StubSynthesisJobClient(["inProgress"]))

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-neural").tts(
                text="a" * 3001, voice="Joanna", resp_format="mp3"
            )

        assert excinfo.value.status == 503
        assert fake_clock == [0.5, 1.0, 2.0, 4.0, 5.0], "5s is the interval cap"
        assert timeout <= sum(fake_clock) < timeout + 5.0, (
            "the request ends within one poll interval of the deadline"
        )
        assert client.polls == len(fake_clock) + 1
        assert usage.USAGE.get(), "characters accepted by Polly are billed regardless"
        assert not client.fetched, "an unfinished job has no audio to serve"

        (cleanup,) = pending_cleanups
        await cleanup
        assert client.deleted == [
            ("primary-bucket", f"{SETTINGS.aws_s3_tmp_prefix}speech.{_JOB_ID}.mp3")
        ], "a timed-out job still has its audio object deleted"

    async def test_a_failed_job_never_reports_its_backend_reason(
        self,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
        pending_cleanups: list[Awaitable[None]],
    ) -> None:
        """The failure reason stays in the log; the caller gets a clean message."""
        client = stub_synthesis_job(
            _StubSynthesisJobClient(
                ["failed"], reason="Access denied writing to s3://internal-bucket"
            )
        )

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-neural").tts(
                text="a" * 3001, voice="Joanna", resp_format="mp3"
            )

        assert excinfo.value.status == 503
        assert "s3" not in str(excinfo.value).lower()
        assert "denied" not in str(excinfo.value).lower()
        log = REQUEST_LOG.get()
        assert any("Access denied" in str(detail) for detail in log["error_detail"]), (
            "the failure reason must still be logged server-side"
        )

        (cleanup,) = pending_cleanups
        await cleanup
        assert client.deleted == [
            ("primary-bucket", f"{SETTINGS.aws_s3_tmp_prefix}speech.{_JOB_ID}.mp3")
        ], "a failed job may still have written a partial object"

    @pytest.mark.parametrize("length", [3001, 100_001], ids=["above_call", "above_job"])
    async def test_without_a_bucket_the_error_states_the_input_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
        length: int,
    ) -> None:
        """A deployment with no bucket rejects long input, naming its length limit.

        Both bands answer the same: the 100,000-character job limit is only
        reachable where a job can run, so an input beyond it must not be
        rejected with a length this deployment would never accept either.
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)
        client = stub_synthesis_job(_StubSynthesisJobClient(["completed"]))

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-neural").tts(
                text="a" * length, voice="Joanna", resp_format="mp3"
            )

        message = str(excinfo.value)
        assert excinfo.value.status == 400
        assert "3,000" in message, "the caller must be told the length it may send"
        assert "100,000" not in message, "the job limit is not this server's limit"
        assert "aws_s3_bucket" not in message
        assert "s3" not in message.lower()
        assert not client.requests, "Polly must not be asked to do the impossible"
        log = REQUEST_LOG.get()
        assert any("aws_s3_bucket" in str(detail) for detail in log["error_detail"]), (
            "the operator must find the setting in the log"
        )

    async def test_an_unusable_bucket_answers_like_a_missing_one(
        self,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
    ) -> None:
        """A bucket Polly rejects is a deployment fault, not a 502 to retry."""
        stub_synthesis_job(
            _StubSynthesisJobClient(
                ["completed"],
                start_error=ClientError(
                    {
                        "Error": {
                            "Code": "InvalidS3BucketException",
                            "Message": "The bucket 'primary-bucket' is invalid",
                        }
                    },
                    "StartSpeechSynthesisTask",
                ),
            )
        )

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-neural").tts(
                text="a" * 3001, voice="Joanna", resp_format="mp3"
            )

        assert excinfo.value.status == 400
        assert "3,000" in str(excinfo.value)
        assert "primary-bucket" not in str(excinfo.value)
        log = REQUEST_LOG.get()
        assert any("primary-bucket" in str(detail) for detail in log["error_detail"]), (
            "the rejected bucket must still be named in the log"
        )

    async def test_a_denied_start_is_not_disguised_as_a_length_limit(
        self,
        api_key: str,
        monkeypatch: pytest.MonkeyPatch,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
    ) -> None:
        """A denied task start is reported as a denial, not as an input-length limit.

        Only the two storage errors mean "this deployment cannot store the
        audio". A missing ``polly:StartSpeechSynthesisTask`` permission is
        answered as a feature the deployment cannot run, so an operator whose
        IAM policy is incomplete is not told to split their text, and the caller
        is not blamed for a policy only that operator can fix.

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_StartSpeechSynthesisTask.html
             stdapi/models/audio/amazon_polly.py:_synthesize_long_text
             stdapi/api_errors.py:denied_feature_unavailable
        """
        from stdapi.main import app  # noqa: PLC0415
        from stdapi.routes import openai_audio_speech  # noqa: PLC0415

        async def _resolve(model_id: str, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            """Resolve the model without the live Bedrock catalog."""
            return make_model_details(model_id, output_modalities=["SPEECH"])

        monkeypatch.setattr(openai_audio_speech, "validate_model", _resolve)
        client = stub_synthesis_job(
            _StubSynthesisJobClient(
                ["completed"],
                start_error=make_client_error(
                    "AccessDeniedException",
                    "StartSpeechSynthesisTask",
                    message=(
                        "User: arn:aws:sts::123456789012:assumed-role/gw is not "
                        "authorized to perform: polly:StartSpeechSynthesisTask"
                    ),
                    status=403,
                ),
            )
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://gateway",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as http_client:
            response = await http_client.post(
                "/v1/audio/speech",
                json={
                    "model": "amazon.polly-neural",
                    "voice": "Joanna",
                    "input": "a" * 3001,
                },
            )

        assert response.status_code == 503, response.text
        error = response.json()["error"]
        assert error["code"] == "feature_unavailable"
        assert "3,000" not in error["message"], (
            "a permission failure must not be reported as an input too long"
        )
        assert "polly" not in error["message"].lower()
        assert not client.fetched, "a task that never started has no audio to serve"


def _billed_characters() -> int:
    """Return the character count of the request's single Polly usage record."""
    (record,) = usage.USAGE.get().values()
    return record.quantities[Dimension.INPUT_CHARACTERS]


class _FakeBidiPollyClient:
    """Fake bidirectional Polly client returning one scripted duplex stream."""

    def __init__(self, stream: FakeDuplexStream) -> None:
        self.stream = stream
        self.inputs: list[Any] = []

    async def start_speech_synthesis_stream(self, input: Any) -> FakeDuplexStream:  # noqa: A002, ANN401
        """Record the synthesis parameters and hand out the scripted stream."""
        self.inputs.append(input)
        return self.stream


def _audio(payload: bytes) -> StartSpeechSynthesisStreamEventStreamAudioEvent:
    """Build one audio event carrying *payload*."""
    return StartSpeechSynthesisStreamEventStreamAudioEvent(
        AudioEvent(audio_chunk=payload)
    )


def _closed(characters: int) -> StartSpeechSynthesisStreamEventStreamStreamClosedEvent:
    """Build the trailing event Polly bills the session with."""
    return StartSpeechSynthesisStreamEventStreamStreamClosedEvent(
        StreamClosedEvent(request_characters=characters)
    )


def _validation_error(message: str) -> ValidationException:
    """Build the modeled rejection a stream fails with."""
    return ValidationException(
        message=message, reason=ValidationExceptionReason("other")
    )


@pytest.fixture
async def stub_bidi_polly(
    monkeypatch: pytest.MonkeyPatch, _request_context: None
) -> Callable[[FakeDuplexStream], _FakeBidiPollyClient]:
    """Register generative ``Joanna`` in both regions, with no bucket anywhere.

    Returns:
        Factory binding a fake bidirectional client serving *stream* in every
        region, and returning it for request assertions.
    """
    _patch_voices(
        monkeypatch,
        {
            ("generative", "us-east-1"): {"Joanna"},
            ("generative", "eu-west-1"): {"Joanna"},
        },
    )
    await initialize_polly_models()
    amazon_polly._VOICES_BY_NAME_LOWER["joanna"] = "Joanna"  # noqa: SLF001
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)
    monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})

    def _bind(stream: FakeDuplexStream) -> _FakeBidiPollyClient:
        client = _FakeBidiPollyClient(stream)
        monkeypatch.setattr(
            stdapi.aws_bidi,
            "_BIDI_CLIENTS",
            {"polly": {"us-east-1": client, "eu-west-1": client}},
        )
        return client

    return _bind


class TestStreamedSynthesis:
    """AudioModel.tts: a long generative input is synthesized incrementally.

    StartSpeechSynthesisStream returns audio as it is produced, needs no
    storage, and reports what it billed only in its closing event. It accepts
    the generative engine alone and cannot produce speech marks, so every
    other long input keeps the path it had.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/bidirectional-streaming-lifecycle.html
         https://docs.aws.amazon.com/polly/latest/APIReference/API_StartSpeechSynthesisStream.html
         stdapi/models/audio/amazon_polly.py:_synthesize_streamed_text
    """

    async def test_audio_is_streamed_and_billed_from_the_closing_event(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """Every audio chunk is served, and the session's own count is billed."""
        text = "hello world. " * 300  # 3,900 characters
        stub_bidi_polly(
            FakeDuplexStream(
                events=[_audio(b"chunk-1"), _audio(b"chunk-2"), _closed(len(text))]
            )
        )

        response = await get_audio_model("amazon.polly-generative").tts(
            text=text, voice="Joanna", resp_format="mp3"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        assert audio == b"chunk-1chunk-2"
        (key,) = usage.USAGE.get()
        assert key.model == "amazon.polly-generative"
        assert _billed_characters() == len(text)
        assert response["input_tokens"] == len(text)

    async def test_the_synthesis_parameters_travel_in_the_stream_request(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """Voice, engine, format and the extra parameters open the stream."""
        client = stub_bidi_polly(FakeDuplexStream(events=[_closed(3001)]))

        response = await get_audio_model("amazon.polly-generative").tts(
            text="a" * 3001,
            voice="Joanna",
            resp_format="ogg",
            extra_params={"LexiconNames": ["mylex"], "SampleRate": 16000},
        )
        await response["audio_stream"].aclose()

        (request,) = client.inputs
        assert request.engine == "generative"
        assert request.voice_id == "Joanna"
        assert request.output_format == "ogg_vorbis"
        assert request.sample_rate == "16000"
        assert request.lexicon_names == ["mylex"]

    async def test_the_text_is_sent_then_the_input_half_closed(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """Polly ends a session on a close event followed by an input half-close.

        Without the half-close the service keeps waiting for the next input
        event and drops the session five seconds later, truncating the audio.
        """
        text = "hello world. " * 500  # 6,500 characters
        stream = FakeDuplexStream(events=[_audio(b"a"), _closed(len(text))])
        stub_bidi_polly(stream)

        response = await get_audio_model("amazon.polly-generative").tts(
            text=text, voice="Joanna", resp_format="mp3"
        )
        assert b"".join([chunk async for chunk in response["audio_stream"]]) == b"a"

        sent = stream.input_stream.sent
        assert len(sent) == 4, "three text events and one close event"
        assert "".join(event.value.text for event in sent[:-1]) == text
        assert isinstance(
            sent[-1], StartSpeechSynthesisStreamActionStreamCloseStreamEvent
        )
        assert stream.input_stream.closed >= 1, "the input stream must be half-closed"

    async def test_a_failure_after_the_first_chunk_still_bills_what_was_sent(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """A failed session sends no closing event, yet Polly billed the text."""
        text = "a" * 3001
        stub_bidi_polly(
            FakeDuplexStream(
                events=[_audio(b"partial")],
                receive_error=_validation_error("Invalid SSML request"),
            )
        )

        response = await get_audio_model("amazon.polly-generative").tts(
            text=text, voice="Joanna", resp_format="mp3"
        )
        audio_stream = response["audio_stream"]
        delivered = await anext(audio_stream)
        with pytest.raises(ApiError) as excinfo:
            await anext(audio_stream)

        assert delivered == b"partial", "audio produced before the failure is delivered"
        assert excinfo.value.status == 400
        assert "Invalid SSML" not in str(excinfo.value), (
            "the backend message must not reach the caller"
        )
        assert _billed_characters() == len(text)

    async def test_text_that_could_not_be_sent_ends_the_response_in_an_error(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """Audio missing its text is a failure, never a shorter recording.

        The service closes the session on the text it did receive, so the
        response would otherwise end cleanly on a fraction of the input.
        """
        stub_bidi_polly(
            FakeDuplexStream(
                events=[_audio(b"partial"), _closed(11)],
                send_error=_validation_error("Invalid inbound event"),
            )
        )

        response = await get_audio_model("amazon.polly-generative").tts(
            text="a" * 3001, voice="Joanna", resp_format="mp3"
        )
        with pytest.raises(ApiError) as excinfo:
            assert [chunk async for chunk in response["audio_stream"]]

        assert excinfo.value.status == 400
        assert usage.USAGE.get(), "the characters the service did accept are billed"

    async def test_a_client_leaving_early_still_bills_the_session(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """Abandoning the audio does not cancel what Polly already synthesized."""
        text = "a" * 3001
        stub_bidi_polly(FakeDuplexStream(events=[_audio(b"first"), _audio(b"second")]))

        response = await get_audio_model("amazon.polly-generative").tts(
            text=text, voice="Joanna", resp_format="mp3"
        )
        stream = response["audio_stream"]
        assert await anext(stream) == b"first"
        await stream.aclose()

        assert _billed_characters() == len(text)

    async def test_a_rejected_stream_is_a_caller_error_not_a_broken_response(
        self, stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient]
    ) -> None:
        """A request the service refuses fails before any byte is promised."""
        stub_bidi_polly(
            FakeDuplexStream(open_error=_validation_error("Invalid VoiceId parameter"))
        )

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-generative").tts(
                text="a" * 3001, voice="Joanna", resp_format="mp3"
            )

        assert excinfo.value.status == 400
        assert "VoiceId" not in str(excinfo.value)
        assert not usage.USAGE.get(), "nothing was synthesized"

    async def test_a_stream_that_cannot_open_falls_back_to_a_job(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient],
        pending_cleanups: list[Awaitable[None]],
        fake_clock: list[float],
    ) -> None:
        """Where a job can run, an unavailable stream is not a failed request.

        A deployment upgraded without the new permission keeps working: the
        request is served the way it was before, once.
        """
        stub_bidi_polly(
            FakeDuplexStream(open_error=_validation_error("not authorized"))
        )
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "primary-bucket")
        monkeypatch.setattr(
            stdapi.aws_s3, "BUCKET_TO_REGION", {"primary-bucket": "us-east-1"}
        )
        job_client = _StubSynthesisJobClient(["completed"])

        def _get_client(service: str, region: str | None = None) -> object:
            job_client.clients.append((service, region))
            return job_client

        for module in (amazon_polly, stdapi.aws, stdapi.aws_s3):
            monkeypatch.setattr(module, "get_client", _get_client)

        response = await get_audio_model("amazon.polly-generative").tts(
            text="a" * 3001, voice="Joanna", resp_format="mp3"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        assert audio == b"long-audio"
        assert len(job_client.requests) == 1
        assert fake_clock == []
        assert len(usage.USAGE.get()) == 1, "the abandoned stream must bill nothing"
        assert _billed_characters() == 3001, "the job's own count is the billed one"
        for cleanup in pending_cleanups:
            await cleanup

    async def test_a_non_generative_long_input_is_never_streamed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_synthesis_job: Callable[
            [_StubSynthesisJobClient], _StubSynthesisJobClient
        ],
        pending_cleanups: list[Awaitable[None]],
        fake_clock: list[float],
    ) -> None:
        """Only the generative engine streams, so the others keep the job path."""
        stream = FakeDuplexStream(events=[_closed(3001)])
        monkeypatch.setattr(
            stdapi.aws_bidi,
            "_BIDI_CLIENTS",
            {"polly": {"us-east-1": _FakeBidiPollyClient(stream)}},
        )
        client = stub_synthesis_job(_StubSynthesisJobClient(["completed"]))

        response = await get_audio_model("amazon.polly-neural").tts(
            text="a" * 3001, voice="Joanna", resp_format="mp3"
        )
        audio = b"".join([chunk async for chunk in response["audio_stream"]])

        assert audio == b"long-audio"
        assert client.requests, "the job path must still serve the request"
        assert not stream.input_stream.sent, "no stream may have been opened"
        for cleanup in pending_cleanups:
            await cleanup

    async def test_speech_marks_are_never_streamed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient],
    ) -> None:
        """Timing marks are not an audio format a stream can produce.

        The stream's ``OutputFormat`` has no ``json`` member, so a speech-marks
        request that is too long for one call is rejected with the length it
        accepts rather than silently served as audio.
        """
        stream = FakeDuplexStream(events=[_closed(3001)])
        stub_bidi_polly(stream)

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-generative").tts(
                text="a" * 3001,
                voice="Joanna",
                resp_format="mp3",
                extra_params={"SpeechMarkTypes": ["word"]},
            )

        assert excinfo.value.status == 400
        assert "3,000" in str(excinfo.value)
        assert not stream.input_stream.sent

    async def test_a_caller_ssml_document_is_never_split_across_events(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_bidi_polly: Callable[[FakeDuplexStream], _FakeBidiPollyClient],
    ) -> None:
        """An SSML document must be self-contained, so a long one cannot stream."""
        stream = FakeDuplexStream(events=[_closed(3001)])
        stub_bidi_polly(stream)
        document = f"<speak>{'a' * 3001}</speak>"

        with pytest.raises(ApiError) as excinfo:
            await get_audio_model("amazon.polly-generative").tts(
                text=document, voice="Joanna", resp_format="mp3"
            )

        assert excinfo.value.status == 400
        assert not stream.input_stream.sent
