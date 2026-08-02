"""Unit tests for the AWS Translate wrapper: failover, language codes, extra params.

Also covers subtitle translation, which batches every cue of an SRT/VTT file into
a single HTML document so one ``TranslateText`` call serves the whole file, then
puts the translations back without touching cue numbers or timings.

Ref: https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html
     https://docs.aws.amazon.com/translate/latest/dg/translating-html.html
     stdapi/aws_translate.py:translate
     stdapi/aws_translate.py:translate_subtitle
"""

from re import DOTALL
from re import compile as compile_regex

import pytest
from botocore.exceptions import ParamValidationError

import stdapi.aws
from stdapi import usage
from stdapi.api_errors import ApiError
from stdapi.aws_translate import translate, translate_subtitle
from stdapi.config import SETTINGS
from stdapi.pricing import Dimension, Service
from tests._helpers import make_client_error

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Reads the segment number and inner text of each span the wrapper sends out.
_SPAN_RE = compile_regex(r'<span id="seg(\d+)">(.*?)</span>', DOTALL)


class _StubTranslateClient:
    """Stub Translate client with a fixed per-region behavior."""

    def __init__(self, outcome: dict[str, str] | Exception) -> None:
        self._outcome = outcome
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    async def translate_text(self, **kwargs: object) -> dict[str, str]:
        """Record the request and return the fixed payload or raise the error."""
        self.calls += 1
        self.requests.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


@pytest.fixture(autouse=True)
def _two_candidate_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to two candidate regions with no explicit Translate region."""
    monkeypatch.setattr(SETTINGS, "aws_translate_region", None)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])


def _patch_clients(
    monkeypatch: pytest.MonkeyPatch, clients: dict[str, _StubTranslateClient]
) -> None:
    monkeypatch.setattr(
        stdapi.aws, "get_client", lambda _service, region=None: clients[region]
    )


class TestTranslateFailover:
    """translate(): ordered failover across candidate regions.

    Ref: stdapi/aws.py:call_with_region_failover
         stdapi/aws.py:is_failover_error
    """

    @pytest.mark.usefixtures("usage_scope")
    async def test_failover_serves_and_records_the_second_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first region falls over; usage records the second.

        Billing must follow the region that actually served the call, so exactly
        one usage record is kept and it carries the second region and the input
        character count Translate charges on.
        """
        clients = {
            "us-east-1": _StubTranslateClient(
                make_client_error("ThrottlingException", "TranslateText")
            ),
            "eu-west-1": _StubTranslateClient({"TranslatedText": "hello"}),
        }
        _patch_clients(monkeypatch, clients)

        assert await translate("bonjour", "fr") == "hello"

        assert clients["us-east-1"].calls == 1
        assert clients["eu-west-1"].calls == 1
        records = usage.USAGE.get()
        (key,) = records
        assert key.region == "eu-west-1", (
            "usage must not be billed to the failed region"
        )
        assert key.service == Service.TRANSLATE
        assert key.model == "amazon.translate"
        assert records[key].quantities == {Dimension.INPUT_CHARACTERS: len("bonjour")}

    async def test_unsupported_language_pair_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A language-pair error is a caller error: 400, single region tried.

        ``UnsupportedLanguagePairException`` would fail identically everywhere,
        so it must abort the failover loop instead of consuming every region.
        The raw AWS error code is not forwarded to the client.
        """
        clients = {
            "us-east-1": _StubTranslateClient(
                make_client_error("UnsupportedLanguagePairException", "TranslateText")
            ),
            "eu-west-1": _StubTranslateClient({"TranslatedText": "x"}),
        }
        _patch_clients(monkeypatch, clients)

        with pytest.raises(ApiError, match="not supported") as excinfo:
            await translate("hej", "xx")

        assert excinfo.value.status == 400
        assert "UnsupportedLanguagePairException" not in str(excinfo.value)
        assert clients["us-east-1"].calls == 1
        assert clients["eu-west-1"].calls == 0

    async def test_english_source_short_circuits_without_any_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An English source returns the text untouched, no AWS call.

        Amazon Translate does not bill same-language pairs, and the translation
        route always targets English, so an English transcript must skip the call.

        Ref: https://docs.aws.amazon.com/translate/latest/dg/what-is-languages.html
        """
        clients = {
            "us-east-1": _StubTranslateClient({"TranslatedText": "never"}),
            "eu-west-1": _StubTranslateClient({"TranslatedText": "never"}),
        }
        _patch_clients(monkeypatch, clients)

        assert await translate("hello", "en-US") == "hello"

        assert clients["us-east-1"].calls == 0
        assert clients["eu-west-1"].calls == 0


class TestTranslateSourceLanguageCode:
    """translate(): region subtags are stripped, except distinct Translate variants.

    Amazon Transcribe reports full locales (``fr-FR``) while Amazon Translate takes
    ISO 639-1 codes with RFC 5646 variants for only five languages, so every other
    subtag must be dropped before the call.

    Ref: https://docs.aws.amazon.com/translate/latest/dg/what-is-languages.html
         stdapi/aws_translate.py:_TRANSLATE_DISTINCT_LANGUAGE_CODES
    """

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("fr-FR", "fr"),
            ("fr-CA", "fr-CA"),
            ("es-US", "es"),
            ("es-MX", "es-MX"),
            ("zh-CN", "zh"),
            ("zh-TW", "zh-TW"),
            ("pt-PT", "pt-PT"),
            ("pt-BR", "pt"),
            ("fa-IR", "fa"),
            ("fa-AF", "fa-AF"),
        ],
    )
    async def test_source_language_code_sent_to_translate(
        self, monkeypatch: pytest.MonkeyPatch, source: str, expected: str
    ) -> None:
        """Distinct Translate variants keep their region subtag; others are stripped."""
        client = _StubTranslateClient({"TranslatedText": "hi"})
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )

        await translate("bonjour", source)

        (request,) = client.requests
        assert request["SourceLanguageCode"] == expected
        assert request["TargetLanguageCode"] == "en"
        assert request["Text"] == "bonjour"


class TestTranslateExtraParams:
    """translate(): Settings/TerminologyNames extra parameters (issue #85).

    Ref: https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html
         stdapi/models/audio/amazon_transcribe.py:_pop_translate_extra_params
    """

    async def test_settings_and_terminology_names_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings and TerminologyNames land in the TranslateText request.

        Values are forwarded verbatim: Amazon Translate silently ignores a
        ``Formality`` its target language does not support instead of erroring,
        so the wrapper must not pre-filter them.

        Ref: https://docs.aws.amazon.com/translate/latest/dg/customizing-translations-formality.html
             https://docs.aws.amazon.com/translate/latest/dg/customizing-translations-profanity.html
        """
        client = _StubTranslateClient({"TranslatedText": "hi"})
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )

        await translate(
            "bonjour",
            "fr",
            settings={"Formality": "FORMAL", "Profanity": "MASK"},
            terminology_names=["MyGlossary"],
        )

        (request,) = client.requests
        assert request["Settings"] == {"Formality": "FORMAL", "Profanity": "MASK"}
        assert request["TerminologyNames"] == ["MyGlossary"]

    async def test_omitted_extra_params_are_not_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without settings/terminology_names, the request carries neither key."""
        client = _StubTranslateClient({"TranslatedText": "hi"})
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )

        await translate("bonjour", "fr")

        (request,) = client.requests
        assert "Settings" not in request
        assert "TerminologyNames" not in request

    @pytest.mark.usefixtures("request_log")
    async def test_param_validation_error_becomes_a_400(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid Settings/TerminologyNames value is a caller 400, not a 500.

        ``ParamValidationError`` is a ``BotoCoreError``, so the failover loop
        treats it as region-level and exhausts every candidate before the last
        one's error is converted. The raw botocore field-path report is logged
        server-side only; the client gets a generic message.

        Ref: https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html
             stdapi/aws.py:is_failover_error
        """
        clients = {
            "us-east-1": _StubTranslateClient(
                ParamValidationError(report="Invalid Formality value")
            ),
            "eu-west-1": _StubTranslateClient(
                ParamValidationError(report="Invalid Formality value")
            ),
        }
        _patch_clients(monkeypatch, clients)

        with pytest.raises(ApiError, match="Invalid translation settings") as excinfo:
            await translate("bonjour", "fr", settings={"Formality": "BAD"})

        assert excinfo.value.status == 400, "a client-side rejection must not be a 500"
        assert excinfo.value.code is None
        assert "Invalid Formality value" not in str(excinfo.value)
        assert clients["us-east-1"].calls == 1
        assert clients["eu-west-1"].calls == 1


class _StubSubtitleTranslateClient:
    """Translate stub that brackets each span and returns them in reverse order.

    Amazon Translate is free to reorder or re-wrap the spans it returns, so the
    stub emits the last segment first: a reader that trusted document order rather
    than the ``seg<n>`` ID would swap the cues.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def translate_text(self, **kwargs: object) -> dict[str, str]:
        """Record the request and echo every span back, bracketed and reversed."""
        self.requests.append(kwargs)
        spans = _SPAN_RE.findall(str(kwargs["Text"]))
        body = "".join(
            f'<span id="seg{index}">[{text}]</span>' for index, text in reversed(spans)
        )
        return {"TranslatedText": f"<!DOCTYPE html><html><body>{body}</body></html>"}


#: Three-cue SRT sample: single-line, two-line, and markup-bearing cue text.
_SRT = """1
00:00:01,000 --> 00:00:03,000
Bonjour tout le monde

2
00:00:04,000 --> 00:00:06,000
Comment ca va ?
Tres bien

3
00:00:07,000 --> 00:00:09,000
Tom & Jerry <3
"""

#: WebVTT sample whose header must survive untranslated.
_VTT = """WEBVTT

1
00:00:01.000 --> 00:00:03.000
Hola mundo
"""


class TestTranslateSubtitle:
    """translate_subtitle(): cue text is translated, everything else is preserved.

    Ref: https://docs.aws.amazon.com/translate/latest/dg/translating-html.html
         stdapi/aws_translate.py:_subtitle_extract_text_segments
         stdapi/aws_translate.py:_subtitle_parse_translated_html
         stdapi/aws_translate.py:_subtitle_reconstruct_with_translation
    """

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> _StubSubtitleTranslateClient:
        """Serve the same subtitle stub from every candidate region.

        Returns:
            The stub client every region resolves to.
        """
        client = _StubSubtitleTranslateClient()
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )
        return client

    async def test_srt_cue_text_is_replaced_and_the_structure_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cue numbers, timings and blank lines survive; only the text lines change.

        A single call carries all three cues, a two-line cue stays one segment (so
        the translator sees the sentence whole), and the segments are re-applied by
        span ID even though the stub returns them last-first.
        """
        client = self._patch(monkeypatch)

        result = await translate_subtitle(_SRT, "fr-FR")

        assert (
            result
            == """1
00:00:01,000 --> 00:00:03,000
[Bonjour tout le monde]

2
00:00:04,000 --> 00:00:06,000
[Comment ca va ?
Tres bien]

3
00:00:07,000 --> 00:00:09,000
[Tom & Jerry <3]
"""
        )
        assert len(client.requests) == 1, "the whole file must cost one call"

    async def test_cue_text_is_html_escaped_on_the_wire_and_unescaped_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Markup characters in a cue are escaped into the span and restored after.

        Sending ``<3`` raw would make Translate treat it as a tag and drop it from
        the returned document; leaving the entities in place would surface
        ``&amp;`` to the end user.
        """
        client = self._patch(monkeypatch)

        result = await translate_subtitle(_SRT, "fr-FR")

        (request,) = client.requests
        assert "Tom &amp; Jerry &lt;3" in str(request["Text"])
        assert "Tom & Jerry <3" not in str(request["Text"])
        assert "[Tom & Jerry <3]" in result
        assert "&amp;" not in result

    async def test_webvtt_header_and_cue_numbers_are_never_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only cue text reaches Translate: the WEBVTT header and numbers stay put.

        Both are structural, and a translated ``WEBVTT`` line makes the file
        unparseable for every player.

        Ref: https://www.w3.org/TR/webvtt1/#file-structure
        """
        client = self._patch(monkeypatch)

        result = await translate_subtitle(_VTT, "es-US")

        (request,) = client.requests
        assert _SPAN_RE.findall(str(request["Text"])) == [("0", "Hola mundo")]
        assert (
            result
            == """WEBVTT

1
00:00:01.000 --> 00:00:03.000
[Hola mundo]
"""
        )

    async def test_content_without_cue_text_is_returned_without_any_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A header-only file is returned unchanged and costs nothing.

        Amazon Translate bills per input character, so a file with no cue text must
        short-circuit rather than send an empty HTML document.
        """
        client = self._patch(monkeypatch)

        assert await translate_subtitle("WEBVTT\n\n", "es-US") == "WEBVTT\n\n"

        assert client.requests == []
