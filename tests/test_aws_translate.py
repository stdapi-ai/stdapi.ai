"""Unit tests for AWS Translate multi-region failover."""

from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, ParamValidationError

import stdapi.aws
from stdapi import usage
from stdapi.api_errors import ApiError
from stdapi.aws_translate import translate
from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Generator


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


class _StubTranslateClient:
    """Stub Translate client with a fixed per-region behavior."""

    def __init__(self, outcome: dict[str, str] | Exception) -> None:
        self._outcome = outcome
        self.calls = 0
        self.requests: list[dict[str, str]] = []

    async def translate_text(self, **kwargs: str) -> dict[str, str]:
        """Record the request and return the fixed payload or raise the error."""
        self.calls += 1
        self.requests.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _client_error(code: str) -> ClientError:
    response: Any = {"Error": {"Code": code, "Message": code}}
    return ClientError(response, "TranslateText")


@pytest.fixture(autouse=True)
def _two_candidate_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to two candidate regions with no explicit Translate region."""
    monkeypatch.setattr(SETTINGS, "aws_translate_region", None)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])


@pytest.fixture
def _usage_scope() -> Generator[None]:
    """Install a fresh usage accumulator for the test."""
    token = usage.init_usage()
    yield
    usage.USAGE.reset(token)


def _patch_clients(
    monkeypatch: pytest.MonkeyPatch, clients: dict[str, _StubTranslateClient]
) -> None:
    monkeypatch.setattr(
        stdapi.aws, "get_client", lambda _service, region=None: clients[region]
    )


class TestTranslateFailover:
    """translate(): ordered failover across candidate regions."""

    @pytest.mark.usefixtures("_usage_scope")
    async def test_failover_serves_and_records_the_second_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first region falls over; usage records the second."""
        clients = {
            "us-east-1": _StubTranslateClient(_client_error("ThrottlingException")),
            "eu-west-1": _StubTranslateClient({"TranslatedText": "hello"}),
        }
        _patch_clients(monkeypatch, clients)

        assert await translate("bonjour", "fr") == "hello"

        assert clients["us-east-1"].calls == 1
        assert clients["eu-west-1"].calls == 1
        (key,) = usage.USAGE.get()
        assert key.region == "eu-west-1"

    async def test_unsupported_language_pair_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A language-pair error is a caller error: 400, single region tried."""
        clients = {
            "us-east-1": _StubTranslateClient(
                _client_error("UnsupportedLanguagePairException")
            ),
            "eu-west-1": _StubTranslateClient({"TranslatedText": "x"}),
        }
        _patch_clients(monkeypatch, clients)

        with pytest.raises(ApiError, match="not supported"):
            await translate("hej", "xx")

        assert clients["us-east-1"].calls == 1
        assert clients["eu-west-1"].calls == 0

    async def test_english_source_short_circuits_without_any_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An English source returns the text untouched, no AWS call."""
        clients = {
            "us-east-1": _StubTranslateClient({"TranslatedText": "never"}),
            "eu-west-1": _StubTranslateClient({"TranslatedText": "never"}),
        }
        _patch_clients(monkeypatch, clients)

        assert await translate("hello", "en-US") == "hello"

        assert clients["us-east-1"].calls == 0


class TestTranslateSourceLanguageCode:
    """translate(): region subtags are stripped, except distinct Translate variants."""

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


class TestTranslateExtraParams:
    """translate(): Settings/TerminologyNames extra parameters (issue #85)."""

    async def test_settings_and_terminology_names_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings and TerminologyNames land in the TranslateText request."""
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

    async def test_param_validation_error_becomes_a_400(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid Settings/TerminologyNames value is a caller 400, not a 500.

        Both candidate regions raise the same client-side validation error
        (it depends only on the request, not on which region serves it), so
        it is not masked by cross-region failover.
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

        with pytest.raises(ApiError, match="Invalid Formality value"):
            await translate("bonjour", "fr", settings={"Formality": "BAD"})
