"""API-key authentication in :mod:`stdapi.auth`.

``AuthenticationHandler.initialize`` returns whether authentication ends up
enabled, and that boolean is what decides between a locked-down gateway and an
open one, so an empty value from any source must not be mistaken for a key.

Ref: stdapi/auth.py:AuthenticationHandler
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest
from botocore.exceptions import ClientError
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

import stdapi.auth
from stdapi.api_errors import ApiError
from stdapi.auth import AuthenticationHandler, authenticate, initialize_authentication
from stdapi.config import SETTINGS
from tests._helpers import make_event_log

if TYPE_CHECKING:
    from types import TracebackType

pytestmark = pytest.mark.local


async def test_empty_api_key_disables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty API key (e.g. from an empty secret) is treated as auth disabled.

    With no key hashed, ``verify_credentials`` short-circuits and accepts even a
    request that carries no credentials at all.
    """
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr(""))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    assert await handler.initialize() is False
    handler.verify_credentials(None)


async def test_empty_api_key_from_ssm_disables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty value retrieved from SSM is treated as auth disabled.

    The emptiness check runs after the source is read, so an existing but blank
    SSM parameter behaves exactly like an unset ``api_key``.
    """

    async def _empty_ssm_value() -> SecretStr:
        return SecretStr("")

    monkeypatch.setattr(SETTINGS, "api_key", None)
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", "/stdapi/api-key")
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    monkeypatch.setattr(
        AuthenticationHandler, "_get_api_key_from_ssm", staticmethod(_empty_ssm_value)
    )
    handler = AuthenticationHandler()
    assert await handler.initialize() is False
    handler.verify_credentials(None)


@pytest.mark.usefixtures("request_log")
async def test_nonempty_api_key_enables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty API key enables authentication and rejects bad credentials.

    The plaintext key is dropped from ``SETTINGS`` once hashed, and both a wrong
    token and a missing token are rejected with the same detail-free 401 so the
    response cannot distinguish the two.

    Ref: stdapi/auth.py:verify_credentials
    """
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr("a-real-secret"))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    assert await handler.initialize() is True
    assert SETTINGS.api_key is None, "the plaintext key must not stay in SETTINGS"
    handler.verify_credentials(SecretStr("a-real-secret"))

    with pytest.raises(ApiError) as wrong_key:
        handler.verify_credentials(SecretStr("wrong-secret"))
    assert wrong_key.value.status == 401
    assert str(wrong_key.value) == "Unauthorized"

    with pytest.raises(ApiError) as no_key:
        handler.verify_credentials(None)
    assert no_key.value.status == 401
    assert str(no_key.value) == "Unauthorized"


class _FakeSecretsManagerCM:
    """Async context manager standing in for an aiobotocore ``secretsmanager`` client.

    ``_get_api_key_from_secrets_manager`` opens the client with ``async with``, so the
    stub has to satisfy that protocol rather than being a plain object.
    """

    def __init__(self, secret_string: str | None) -> None:
        self._secret_string = secret_string
        self.secret_ids: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    async def get_secret_value(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803
        """Return the canned ``SecretString``, or raise the not-found ``ClientError``."""
        self.secret_ids.append(SecretId)
        if self._secret_string is None:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}},
                "GetSecretValue",
            )
        return {"SecretString": self._secret_string}


def _stub_secretsmanager(
    monkeypatch: pytest.MonkeyPatch, secret_string: str | None
) -> _FakeSecretsManagerCM:
    """Point ``stdapi.auth``'s AWS session at a canned Secrets Manager response."""
    client = _FakeSecretsManagerCM(secret_string)

    def _create_client(service: str, **_kwargs: object) -> _FakeSecretsManagerCM:
        assert service == "secretsmanager"
        return client

    session = type("_Session", (), {"create_client": staticmethod(_create_client)})()
    monkeypatch.setattr(stdapi.auth, "AWS_SESSION", session)
    monkeypatch.setattr(SETTINGS, "api_key", None)
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", "stdapi/api-key")
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_key", "api_key")
    return client


class TestSecretsManagerApiKeySource:
    """Third API-key source: a JSON secret in AWS Secrets Manager.

    The secret holds a JSON document and ``api_key_secretsmanager_key`` selects the
    field inside it, so both the AWS lookup and the key lookup can fail independently.

    Ref: stdapi/auth.py:AuthenticationHandler._get_api_key_from_secrets_manager
    """

    async def test_secret_json_key_enables_authentication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-empty value under the configured key enables authentication.

        The configured secret name is what is requested from AWS, and the plaintext
        secret name is cleared from ``SETTINGS`` once consumed.
        """
        client = _stub_secretsmanager(monkeypatch, '{"api_key": "s3cr3t"}')
        handler = AuthenticationHandler()

        assert await handler.initialize() is True

        assert client.secret_ids == ["stdapi/api-key"]
        assert SETTINGS.api_key_secretsmanager_secret is None
        handler.verify_credentials(SecretStr("s3cr3t"))

    @pytest.mark.usefixtures("request_log")
    async def test_secret_json_key_rejects_other_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A credential other than the secret's value is rejected with a 401."""
        _stub_secretsmanager(monkeypatch, '{"api_key": "s3cr3t"}')
        handler = AuthenticationHandler()
        assert await handler.initialize() is True

        with pytest.raises(ApiError) as exc_info:
            handler.verify_credentials(SecretStr("wrong-secret"))
        assert exc_info.value.status == 401

    async def test_empty_secret_value_disables_authentication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty value under the configured key leaves the gateway unauthenticated.

        This mirrors the SSM source: emptiness is checked after the source is read, so
        a present-but-blank secret must not be mistaken for a key.
        """
        _stub_secretsmanager(monkeypatch, '{"api_key": ""}')
        handler = AuthenticationHandler()

        assert await handler.initialize() is False

        handler.verify_credentials(None)

    async def test_missing_key_in_secret_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A secret without the configured key fails startup naming key and secret."""
        _stub_secretsmanager(monkeypatch, '{"other": "s3cr3t"}')
        handler = AuthenticationHandler()

        with pytest.raises(ValueError, match="'api_key' not found in secret") as exc:
            await handler.initialize()
        assert "stdapi/api-key" in str(exc.value)

    async def test_missing_secret_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ResourceNotFoundException`` becomes a ``ValueError`` naming the secret.

        A missing secret is an operator configuration error, so it is surfaced as a
        startup failure rather than as a raw botocore exception.
        """
        _stub_secretsmanager(monkeypatch, None)
        handler = AuthenticationHandler()

        with pytest.raises(ValueError, match="'stdapi/api-key' not found"):
            await handler.initialize()


@pytest.mark.usefixtures("request_log")
class TestAuthenticateDependency:
    """authenticate: which header is validated, and what is left behind afterwards.

    A client may send both ``x-api-key`` and ``Authorization: Bearer`` (an Anthropic
    SDK pointed at an OpenAI-prefixed route, for instance); only one of them decides
    the outcome.

    Ref: stdapi/auth.py:authenticate
    """

    @staticmethod
    def _enabled_handler(monkeypatch: pytest.MonkeyPatch) -> None:
        """Install a global handler that only accepts ``good-key``."""
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("good-key"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        handler = AuthenticationHandler()
        handler._hash_api_key(SecretStr("good-key"))  # noqa: SLF001
        monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)

    async def test_x_api_key_wins_over_a_wrong_bearer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid ``x-api-key`` is accepted even alongside an invalid bearer token."""
        self._enabled_handler(monkeypatch)

        await authenticate(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="wrong-key"
            ),
            x_api_key="good-key",
        )

    async def test_wrong_x_api_key_rejects_a_valid_bearer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid ``x-api-key`` fails the request even alongside a valid bearer.

        Swapping the two values against the previous test proves ``x-api-key`` is the
        header actually validated rather than merely one of two accepted candidates.
        """
        self._enabled_handler(monkeypatch)

        with pytest.raises(ApiError) as exc_info:
            await authenticate(
                credentials=HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials="good-key"
                ),
                x_api_key="wrong-key",
            )
        assert exc_info.value.status == 401

    async def test_bearer_credential_is_scrubbed_after_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raw bearer string is blanked on the security object once wrapped.

        FastAPI keeps the credentials object alive for the request, so leaving the
        plaintext token on it would expose it to anything dumping the dependency state.
        """
        self._enabled_handler(monkeypatch)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="good-key"
        )

        await authenticate(credentials=credentials, x_api_key=None)

        assert credentials.credentials == ""

    async def test_no_credentials_at_all_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither header present yields the same detail-free 401 as a wrong key."""
        self._enabled_handler(monkeypatch)

        with pytest.raises(ApiError) as exc_info:
            await authenticate(credentials=None, x_api_key=None)
        assert exc_info.value.status == 401
        assert str(exc_info.value) == "Unauthorized"


class _FakeSsmCM:
    """Async context manager standing in for an aiobotocore ``ssm`` client."""

    def __init__(self, value: str | None, error_code: str | None = None) -> None:
        self._value = value
        self._error_code = error_code
        self.calls: list[dict[str, object]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    async def get_parameter(self, **kwargs: object) -> dict[str, dict[str, str]]:
        """Return the canned parameter, or raise the configured ``ClientError``."""
        self.calls.append(kwargs)
        if self._error_code is not None:
            raise ClientError(
                {"Error": {"Code": self._error_code, "Message": "nope"}}, "GetParameter"
            )
        return {"Parameter": {"Value": self._value or ""}}


def _stub_ssm(
    monkeypatch: pytest.MonkeyPatch, value: str | None, error_code: str | None = None
) -> _FakeSsmCM:
    """Point ``stdapi.auth``'s AWS session at a canned SSM response."""
    client = _FakeSsmCM(value, error_code)

    def _create_client(service: str, **_kwargs: object) -> _FakeSsmCM:
        assert service == "ssm"
        return client

    session = type("_Session", (), {"create_client": staticmethod(_create_client)})()
    monkeypatch.setattr(stdapi.auth, "AWS_SESSION", session)
    monkeypatch.setattr(SETTINGS, "api_key", None)
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", "/stdapi/api-key")
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    return client


class TestSsmApiKeySource:
    """Second API-key source: an SSM Parameter Store parameter.

    The parameter is expected to be a ``SecureString``, so the read has to ask
    for decryption; without it AWS answers with the ciphertext and the gateway
    would happily hash that as the key, locking every client out.

    Ref: https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_GetParameter.html
         stdapi/auth.py:AuthenticationHandler._get_api_key_from_ssm
    """

    async def test_parameter_value_is_read_decrypted_and_enables_authentication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The configured parameter is fetched with decryption and becomes the key.

        Any other credential is rejected, so the stored hash is of the parameter
        value rather than of something that happens to accept everything.
        """
        client = _stub_ssm(monkeypatch, "s3cr3t")
        handler = AuthenticationHandler()

        assert await handler.initialize() is True

        assert client.calls == [{"Name": "/stdapi/api-key", "WithDecryption": True}]
        assert SETTINGS.api_key_ssm_parameter is None
        handler.verify_credentials(SecretStr("s3cr3t"))
        with pytest.raises(ApiError) as exc_info:
            handler.verify_credentials(SecretStr("wrong-secret"))
        assert exc_info.value.status == 401

    async def test_missing_parameter_raises_value_error_naming_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ParameterNotFound`` becomes a startup ``ValueError`` naming the parameter.

        A misconfigured parameter name is an operator error, so it is reported
        as such instead of as a raw botocore exception.
        """
        _stub_ssm(monkeypatch, None, "ParameterNotFound")
        handler = AuthenticationHandler()

        with pytest.raises(ValueError, match=r"'/stdapi/api-key' not found"):
            await handler.initialize()

    async def test_other_client_errors_are_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied read propagates as ``ClientError``, not as a "not found".

        ``AccessDeniedException`` means the task role lacks ``ssm:GetParameter``;
        reporting it as a missing parameter would send the operator after the
        wrong fix, and rewriting it as a ``ValueError`` would lose the AWS code.
        """
        _stub_ssm(monkeypatch, None, "AccessDeniedException")
        handler = AuthenticationHandler()

        with pytest.raises(ClientError) as exc:
            await handler.initialize()
        assert exc.value.response["Error"]["Code"] == "AccessDeniedException"


class TestInitializeAuthenticationWarning:
    """An unauthenticated gateway says so, loudly, in its startup log.

    Leaving every source unset is a valid (development) configuration, so it
    cannot fail startup -- which makes the ``start`` event the only signal an
    operator gets that the deployment is open to anyone who can reach it.

    Ref: stdapi/auth.py:initialize_authentication
         stdapi/monitoring.py:add_server_warning
    """

    async def test_no_configured_source_warns_on_the_start_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no key source, the start event is raised to warning and names all three."""
        monkeypatch.setattr(SETTINGS, "api_key", None)
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        start_event = make_event_log(type="start")

        await initialize_authentication(start_event)

        warnings = start_event["server_warnings"]
        assert len(warnings) == 1
        warning = str(warnings[0])
        assert "SECURITY risk" in warning
        assert "api_key" in warning
        assert "api_key_ssm_parameter" in warning
        assert "api_key_secretsmanager_secret" in warning
        assert start_event["level"] == "warning"

    async def test_a_configured_key_produces_no_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With authentication enabled the start event stays clean at info level."""
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("a-real-secret"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        start_event = make_event_log(type="start")

        await initialize_authentication(start_event)

        assert "server_warnings" not in start_event
        assert start_event["level"] == "info"
