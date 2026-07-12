"""Unit tests for AWS Translate multi-region failover."""

from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

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

    async def translate_text(self, **_kwargs: str) -> dict[str, str]:
        """Return the fixed payload or raise the configured error."""
        self.calls += 1
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
