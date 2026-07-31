"""Unit tests for the AWS Translate wrapper: failover, language codes, extra params.

Ref: https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html
     stdapi/aws_translate.py:translate
"""

import pytest
from botocore.exceptions import ParamValidationError

import stdapi.aws
from stdapi import usage
from stdapi.api_errors import ApiError
from stdapi.aws_translate import translate
from stdapi.config import SETTINGS
from stdapi.pricing import Dimension, Service
from tests._helpers import make_client_error

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


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
