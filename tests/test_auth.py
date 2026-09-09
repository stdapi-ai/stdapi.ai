"""API-key authentication in :mod:`stdapi.auth`.

``AuthenticationHandler.initialize`` returns whether authentication ends up
enabled, and that boolean is what decides between a locked-down gateway and an
open one, so an empty value from any source must not be mistaken for a key.

Ref: stdapi/auth.py:AuthenticationHandler
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self

import pytest
from botocore.exceptions import ClientError
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

import stdapi.auth
from stdapi.api_errors import ApiError
from stdapi.auth import (
    AuthenticationHandler,
    authenticate,
    enforce_tenant_endpoint_scope,
    initialize_authentication,
    verify_credential,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import TENANT, Tenant
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

    def __init__(
        self,
        secret_string: str | None,
        *,
        binary: bool = False,
        error_code: str = "ResourceNotFoundException",
    ) -> None:
        self._secret_string = secret_string
        self._binary = binary
        self._error_code = error_code
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
        """Return the canned secret value, or raise the not-found ``ClientError``.

        A secret stored as binary answers with the response AWS really sends
        for one: every member is optional, and the text member is simply absent.
        """
        self.secret_ids.append(SecretId)
        if self._binary:
            return {"Name": SecretId}
        if self._secret_string is None:
            raise ClientError(
                {"Error": {"Code": self._error_code, "Message": "missing"}},
                "GetSecretValue",
            )
        return {"SecretString": self._secret_string}


def _stub_secretsmanager(
    monkeypatch: pytest.MonkeyPatch,
    secret_string: str | None,
    *,
    binary: bool = False,
    error_code: str = "ResourceNotFoundException",
) -> _FakeSecretsManagerCM:
    """Point ``stdapi.auth``'s AWS session at a canned Secrets Manager response."""
    client = _FakeSecretsManagerCM(secret_string, binary=binary, error_code=error_code)

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
    """Third API-key source: a secret in AWS Secrets Manager.

    Both documented shapes have to work: a JSON document, where
    ``api_key_secretsmanager_key`` selects the field inside it and the AWS lookup
    and the key lookup can fail independently, and a plain string, which is the
    key exactly as stored.

    Ref: stdapi/auth.py:AuthenticationHandler._get_api_key_from_secrets_manager
         https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
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

    async def test_a_denied_read_is_not_reported_as_a_missing_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied read propagates as ``ClientError``, not as a "not found".

        ``AccessDeniedException`` means the task role lacks
        ``secretsmanager:GetSecretValue``; reporting it as a missing secret would
        send the operator after the wrong fix.
        """
        _stub_secretsmanager(monkeypatch, None, error_code="AccessDeniedException")
        handler = AuthenticationHandler()

        with pytest.raises(ClientError) as exc:
            await handler.initialize()
        assert exc.value.response["Error"]["Code"] == "AccessDeniedException"

    @pytest.mark.parametrize(
        "secret_string",
        ["my-plain-key", "sk-1234567890abcdef", "1234567890", '"quoted-key"', "true"],
        ids=["plain", "prefixed", "digits-only", "quoted", "json-keyword"],
    )
    async def test_a_plain_string_secret_is_the_api_key(
        self, monkeypatch: pytest.MonkeyPatch, secret_string: str
    ) -> None:
        """A secret that is not a JSON object is the key, byte for byte.

        The documented plain-string deployment stores the key with no JSON around
        it, so nothing may be parsed out of it: a key of digits keeps its leading
        zeroes, and one that happens to be quoted keeps its quotes.
        """
        _stub_secretsmanager(monkeypatch, secret_string)
        handler = AuthenticationHandler()

        assert await handler.initialize() is True

        handler.verify_credentials(SecretStr(secret_string))

    async def test_a_binary_secret_fails_startup_naming_the_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A secret stored as binary is refused with the secret's name, not a KeyError.

        Neither documented shape can be read out of one, and startup is the only
        moment an operator can be told which secret to re-create as text.
        """
        _stub_secretsmanager(monkeypatch, None, binary=True)
        handler = AuthenticationHandler()

        with pytest.raises(ValueError, match="'stdapi/api-key'") as exc:
            await handler.initialize()
        assert "text" in str(exc.value)

    async def test_a_non_text_value_under_the_key_fails_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON secret whose configured key holds an object names key and secret.

        The value is hashed as text, so anything else is an operator error that
        must be reported as one rather than crashing the hash.
        """
        _stub_secretsmanager(monkeypatch, '{"api_key": {"nested": "s3cr3t"}}')
        handler = AuthenticationHandler()

        with pytest.raises(ValueError, match="'api_key'") as exc:
            await handler.initialize()
        assert "stdapi/api-key" in str(exc.value)


class TestRefusalSeverity:
    """An ordinary refused credential is the client's fault, not an incident.

    Every deployment sees wrong, expired and absent keys daily -- a mistyped
    client key, a stale browser token, an internet scanner. Logged without the
    401 they resolve to, they read as ``critical``, which
    ``docs/operations_logging_monitoring.md`` tells operators to open a bug for
    and which the shipped Terraform module pages them for.

    Ref: stdapi/api_errors.py:unauthorized
         stdapi/monitoring.py:_error_level
         docs/operations_logging_monitoring.md
    """

    async def test_a_wrong_api_key_is_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A key that does not match the deployment's own is a warning."""
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("good-key"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        handler = AuthenticationHandler()
        assert await handler.initialize() is True

        with pytest.raises(ApiError):
            handler.verify_credentials(SecretStr("wrong-key"))

        assert request_log["level"] == "warning"
        assert request_log["error_detail"] == ["Invalid API key"]

    async def test_a_missing_api_key_is_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A request carrying no credential at all is a warning too."""
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("good-key"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        handler = AuthenticationHandler()
        assert await handler.initialize() is True

        with pytest.raises(ApiError):
            handler.verify_credentials(None)

        assert request_log["level"] == "warning"
        assert request_log["error_detail"] == ["Missing API key"]

    async def test_a_credential_the_user_pool_refused_is_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """With a user pool as the only method, its refusal is a warning.

        A credential that is not shaped like a signed token never reaches the
        pool, and must not fall through to the disabled key comparison either.

        Ref: stdapi/auth.py:verify_credential
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(SETTINGS, "tenant_api_keys", False)
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", SimpleNamespace(enabled=True)
        )

        with pytest.raises(ApiError) as raised:
            await verify_credential("opaque-credential")

        assert raised.value.status == 401
        assert str(raised.value) == "Unauthorized"
        assert request_log["level"] == "warning"

    async def test_a_credential_that_is_not_a_tenant_key_is_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """With tenant keys as the only method, anything else is a warning.

        Ref: stdapi/auth.py:verify_credential
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)

        with pytest.raises(ApiError) as raised:
            await verify_credential("not-a-tenant-key")

        assert raised.value.status == 401
        assert str(raised.value) == "Unauthorized"
        assert request_log["level"] == "warning"

    async def test_an_out_of_scope_endpoint_is_a_warning(
        self, request_log: dict[str, Any]
    ) -> None:
        """A tenant reaching an endpoint it is not scoped to is a warning.

        Ref: stdapi/auth.py:enforce_tenant_endpoint_scope
        """
        tenant = Tenant(
            key_id="A" * 16, name="acme", endpoints_allow=("/v1/chat/completions",)
        )
        token = TENANT.set(tenant)
        try:
            with pytest.raises(ApiError) as raised:
                enforce_tenant_endpoint_scope(
                    {"type": "http", "route": SimpleNamespace(path_format="/v1/files")}
                )
        finally:
            TENANT.reset(token)

        assert raised.value.status == 401
        assert str(raised.value) == "Unauthorized"
        assert request_log["level"] == "warning"
        assert "A" * 16 in str(request_log["error_detail"][0])


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
