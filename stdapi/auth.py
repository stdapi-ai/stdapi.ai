"""API key and Amazon Cognito user pool authentication for API endpoints."""

from hashlib import blake2b
from hmac import compare_digest
from secrets import token_bytes
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from fastapi import Depends, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretBytes, SecretStr
from pydantic_core import from_json

from stdapi.api_errors import ApiError, unauthorized
from stdapi.auth_cognito import CognitoAuthenticator
from stdapi.aws import CONFIG
from stdapi.config import AWS_REGION, AWS_SESSION, SETTINGS
from stdapi.exceptions import ServerError
from stdapi.monitoring import PRINCIPAL, TENANT, EventLog, Tenant, add_server_warning
from stdapi.tenant_keys import is_tenant_key, verify_tenant_key

if TYPE_CHECKING:
    from collections.abc import Mapping

#: HTTPBearer security scheme for API key authentication
_authorization_bearer = HTTPBearer(auto_error=False)

#: APIKeyHeader security scheme for x-api-key authentication
_x_api_key = APIKeyHeader(name="x-api-key", auto_error=False)


#: Personalisation separating the key-derivation seed from the stored hash.
_DERIVATION_PERSON = b"stdapi-drv"

#: Routeless request standing in when the dependency is called outside FastAPI.
_DIRECT_CALL_REQUEST = Request({"type": "http"})


class AuthenticationHandler:
    """Handles API key authentication with secure hashing for API endpoints."""

    __slots__ = ("_api_key_hash", "_api_key_salt", "_derivation_seed")

    def __init__(self) -> None:
        """Initialize authentication handler with no cached API key hash."""
        self._api_key_hash: SecretBytes | None = None
        self._api_key_salt: SecretBytes | None = None
        self._derivation_seed: SecretBytes | None = None

    def _hash_api_key(self, api_key: SecretStr) -> None:
        """Hash the API key with a random salt using BLAKE2.

        Args:
            api_key: The plain text API key to hash and store securely.
        """
        self._api_key_salt = SecretBytes(token_bytes(16))
        self._api_key_hash = SecretBytes(
            blake2b(
                api_key.get_secret_value().encode("utf-8"),
                salt=self._api_key_salt.get_secret_value(),
            ).digest()
        )
        # Unsalted: a per-instance seed would break tokens across the load balancer.
        self._derivation_seed = SecretBytes(
            blake2b(
                api_key.get_secret_value().encode("utf-8"), person=_DERIVATION_PERSON
            ).digest()
        )

    def derived_key(self, person: bytes, size: int) -> bytes | None:
        """Derive a key of this deployment's own from the configured API key.

        Args:
            person: Personalisation separating this key from any other derived one.
            size: Length of the derived key, in bytes.

        Returns:
            The derived key, or None when no API key is configured.
        """
        if (seed := self._derivation_seed) is None:
            return None
        return blake2b(
            seed.get_secret_value(), digest_size=size, person=person
        ).digest()

    async def initialize(self) -> bool:
        """Initialize authentication by retrieving and securely hashing the API key.

        Called once during application startup.

        Priority order:
        1. Direct configuration (SETTINGS.api_key)
        2. AWS SSM Parameter Store (SETTINGS.api_key_ssm_parameter)
        3. AWS Secrets Manager (SETTINGS.api_key_secretsmanager_secret)

        Returns:
            True if authentication is enabled, False otherwise.

        Raises:
            ClientError: If there's an error retrieving the API key from AWS services.
            ValueError: If configuration is invalid or API key not found.
        """
        api_key: SecretStr | None = None
        if SETTINGS.api_key:
            api_key = SETTINGS.api_key
            SETTINGS.api_key = None
        elif SETTINGS.api_key_ssm_parameter:
            api_key = await self._get_api_key_from_ssm()
            SETTINGS.api_key_ssm_parameter = None
        elif SETTINGS.api_key_secretsmanager_secret:
            api_key = await self._get_api_key_from_secrets_manager()
            SETTINGS.api_key_secretsmanager_secret = None
        if api_key is not None and api_key.get_secret_value():
            self._hash_api_key(api_key)
            return True
        return False

    @staticmethod
    async def _get_api_key_from_ssm() -> SecretStr:
        """Retrieve API key from AWS SSM Parameter Store.

        Returns:
            The API key string from SSM Parameter Store.

        Raises:
            ClientError: If there's an error retrieving the API key from SSM.
            ValueError: If the SSM parameter is not found.
        """
        # Only called with the setting present; the fallback merely narrows the type.
        parameter = SETTINGS.api_key_ssm_parameter or ""
        async with AWS_SESSION.create_client(
            "ssm", config=CONFIG, region_name=AWS_REGION
        ) as ssm_client:
            try:
                return SecretStr(
                    (
                        await ssm_client.get_parameter(
                            Name=parameter, WithDecryption=True
                        )
                    )["Parameter"]["Value"]
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "ParameterNotFound":
                    msg = f"SSM Parameter '{parameter}' not found"
                    raise ValueError(msg) from exc
                raise

    @staticmethod
    async def _get_api_key_from_secrets_manager() -> SecretStr:
        """Retrieve API key from AWS Secrets Manager.

        Both documented shapes are accepted: a JSON object, where
        ``api_key_secretsmanager_key`` selects the field holding the key, and a
        plain string, which is the key exactly as stored. Anything that is not
        a JSON object is therefore taken literally rather than parsed, so a key
        of digits keeps its leading zeroes and a quoted one keeps its quotes.

        Returns:
            The API key string from Secrets Manager.

        Raises:
            ClientError: If there's an error retrieving the API key from Secrets Manager.
            ValueError: If the secret or key is not found, if the secret holds
                binary data, or if the selected field is not text.
        """
        secret_name = SETTINGS.api_key_secretsmanager_secret
        async with AWS_SESSION.create_client(
            "secretsmanager", config=CONFIG, region_name=AWS_REGION
        ) as secrets_client:
            try:
                secret_value = await secrets_client.get_secret_value(
                    SecretId=secret_name
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "ResourceNotFoundException":
                    msg = f"Secret '{secret_name}' not found"
                    raise ValueError(msg) from exc
                raise
        secret_string = secret_value.get("SecretString")
        if secret_string is None:
            msg = f"Secret '{secret_name}' holds no text value; store it as text"
            raise ValueError(msg)
        try:
            document = from_json(secret_string)
        except ValueError:
            document = None
        # Only a JSON object is read as a document; anything else is the key.
        if not isinstance(document, dict):
            return SecretStr(secret_string)
        key_name = SETTINGS.api_key_secretsmanager_key
        try:
            api_key = document[key_name]
        except KeyError as exc:
            msg = f"Key '{key_name}' not found in secret '{secret_name}'"
            raise ValueError(msg) from exc
        if not isinstance(api_key, str):
            msg = f"Key '{key_name}' of secret '{secret_name}' does not hold text"
            # The configuration is wrong, not a caller's type: startup reports
            # every failure of this source the same way.
            raise ValueError(msg)  # noqa: TRY004
        return SecretStr(api_key)

    @property
    def enabled(self) -> bool:
        """Whether an API key is configured.

        Returns:
            True once an API key was found and hashed at startup.
        """
        return self._api_key_hash is not None and self._api_key_salt is not None

    def matches(self, token: str) -> bool:
        """Whether *token* is the deployment's API key, without deciding anything.

        Lets the dispatcher recognize a deployment key that happens to look
        like another credential kind, so introducing new kinds never breaks a
        key that worked before.

        Args:
            token: The candidate credential.

        Returns:
            True when it matches the configured key; False when it does not or
            no key is configured.
        """
        if self._api_key_hash is None or self._api_key_salt is None:
            return False
        return compare_digest(
            blake2b(
                token.encode("utf-8"), salt=self._api_key_salt.get_secret_value()
            ).digest(),
            self._api_key_hash.get_secret_value(),
        )

    def verify_credentials(self, token: SecretStr | None) -> None:
        """Verify authentication for API endpoints.

        Compares *token* against the cached salted hash in constant time. No-op
        when authentication is disabled, allowing all requests.

        Args:
            token: Authentication token.

        Raises:
            ApiError: 401 if authentication is required but missing/invalid.
        """
        if self._api_key_hash is None or self._api_key_salt is None:
            return

        if token is None:
            unauthorized("Missing API key")

        if not compare_digest(
            blake2b(
                token.get_secret_value().encode("utf-8"),
                salt=self._api_key_salt.get_secret_value(),
            ).digest(),
            self._api_key_hash.get_secret_value(),
        ):
            unauthorized("Invalid API key")


#: Global authentication handler instance
_auth_handler = AuthenticationHandler()

#: Authenticator verifying tokens against the configured Amazon Cognito pool.
_cognito_authenticator = CognitoAuthenticator()


async def initialize_authentication(start_event: EventLog) -> None:
    """Initialize the global authentication handlers.

    Called once during application startup; records a security warning on
    *start_event* if no authentication method is configured at all. A method
    that is configured but does not end up enabled fails startup instead, so a
    misconfiguration never resolves into an open deployment.

    Args:
        start_event: Startup event log to update if authentication is disabled.

    Raises:
        ServerError: If a configured API key source resolves to no key, if a
            configured user pool's signing keys cannot be loaded, or if the
            method ``authentication_mode`` demands is not enabled.
    """
    api_key_configured = bool(
        SETTINGS.api_key
        or SETTINGS.api_key_ssm_parameter
        or SETTINGS.api_key_secretsmanager_secret
    )
    api_key_enabled = await _auth_handler.initialize()
    if api_key_configured and not api_key_enabled:
        msg = (
            "The configured API key source holds an empty API key, which would "
            "leave the deployment accepting every request unauthenticated"
        )
        raise ServerError(msg)
    user_pool_enabled = await _cognito_authenticator.initialize()
    tenant_keys_enabled = SETTINGS.tenant_api_keys
    mode = SETTINGS.authentication_mode
    if (mode == "api_key" and not api_key_enabled and not tenant_keys_enabled) or (
        mode == "cognito" and not user_pool_enabled
    ):
        msg = (
            f"authentication_mode '{mode}' is required, but that authentication "
            "method is not enabled"
        )
        raise ServerError(msg)
    if not api_key_enabled and not user_pool_enabled and not tenant_keys_enabled:
        add_server_warning(
            start_event,
            "SECURITY risk: Authentication is not enabled "
            "('api_key', 'api_key_ssm_parameter', 'api_key_secretsmanager_secret', "
            "'aws_cognito_user_pool_id', 'tenant_api_keys' not set)",
        )


async def authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(_authorization_bearer),
    x_api_key: str | None = Depends(_x_api_key),
    request: Request = _DIRECT_CALL_REQUEST,
) -> None:
    """Verify the request credentials dependency for FastAPI routes.

    A credential shaped like a signed token is verified against the configured
    Amazon Cognito user pool and identifies the caller; one shaped like a
    tenant API key is verified against the tenant records and scopes the
    request; anything else is compared against the API key, which identifies
    the deployment rather than a person and so leaves the request with no
    principal. Both headers carry any kind, ``x-api-key`` taking precedence --
    except that a tenant key in ``x-api-key`` verifies *alongside* a Bearer
    credential rather than instead of it: the tenant key authorizes, the token
    identifies, and both must hold. No-op when no authentication method is
    configured, allowing all requests.

    Args:
        credentials: HTTP Bearer token credentials from the Authorization header.
        x_api_key: API key from the X-API-Key header.
        request: The request, whose matched route the tenant scopes apply to.
            Filled by FastAPI; a direct call without one has no route to test,
            which fails closed for an endpoint-restricted tenant.

    Raises:
        ApiError: 401 if authentication is required but missing/invalid, or if
            a tenant key is not allowed on this endpoint.
    """
    # Cleared first, every request: a nested tool call must not inherit an identity.
    PRINCIPAL.set(None)
    TENANT.set(None)
    bearer: str | None = None
    if credentials is not None:
        bearer = credentials.credentials
        credentials.credentials = ""
    if (
        x_api_key
        and SETTINGS.tenant_api_keys
        and is_tenant_key(x_api_key)
        and not _auth_handler.matches(x_api_key)
    ):
        tenant = await verify_tenant_key(x_api_key)
        if bearer:
            await verify_credential(bearer)
        # Set last: a Bearer credential that is itself a tenant key must verify
        # like any other, never re-scope the request onto its own tenant.
        TENANT.set(tenant)
    else:
        await verify_credential(x_api_key or bearer)
    enforce_tenant_endpoint_scope(request.scope)


def scope_route_path(scope: Mapping[str, Any]) -> str | None:
    """Return the matched route's path template from an ASGI scope.

    Args:
        scope: The connection's ASGI scope, after routing.

    Returns:
        The path template, e.g. ``/v1/chat/completions``, or None when no
        route matched.
    """
    route = scope.get("route")
    return getattr(route, "path_format", None) or getattr(route, "path", None)


def enforce_tenant_endpoint_scope(scope: Mapping[str, Any]) -> None:
    """Refuse the connection when its route is outside the tenant's scope.

    No-op when the current request carries no verified tenant.

    Args:
        scope: The connection's ASGI scope, after routing.

    Raises:
        ApiError: 401 when the tenant restricts endpoints and the matched
            route is not allowed.
    """
    if (tenant := TENANT.get()) is not None:
        _enforce_endpoint_scope(tenant, scope_route_path(scope))


def _enforce_endpoint_scope(tenant: Tenant, path: str | None) -> None:
    """Refuse the request when its route is outside the tenant's scope.

    Args:
        tenant: The verified tenant.
        path: The matched route's path template, if known.

    Raises:
        ApiError: 401 when the tenant restricts endpoints and this one is not
            allowed -- or not known, which fails closed.
    """
    if tenant.endpoints_allow is None and not tenant.endpoints_deny:
        return
    if path is None or not tenant.allows_endpoint(path):
        unauthorized(
            f"Tenant API key '{tenant.key_id}' is not allowed on this endpoint"
        )


async def verify_credential(credential: str | None) -> None:
    """Verify one already-extracted credential, whatever carried it.

    Args:
        credential: The credential the caller presented, if any.

    Raises:
        ApiError: 401 if authentication is required but missing/invalid.
        FeatureUnavailableError: 503 if the credential is a tenant key and the
            tenant records cannot be read; never accepted, never a 401.
    """
    if (
        credential
        and SETTINGS.tenant_api_keys
        and is_tenant_key(credential)
        # The deployment key always wins, whatever it is shaped like: a key
        # that worked before tenant keys were enabled must keep working.
        and not _auth_handler.matches(credential)
    ):
        TENANT.set(await verify_tenant_key(credential))
        return
    if _cognito_authenticator.enabled:
        # Three segments is a signed token; a key shaped like one falls through below.
        if credential and credential.count(".") == 2:
            try:
                PRINCIPAL.set(await _cognito_authenticator.verify(credential))
            except ApiError:
                if not _auth_handler.enabled:
                    raise
            else:
                return
        if not _auth_handler.enabled:
            # A disabled API key comparison accepts anything, so reject here instead.
            unauthorized("Credentials rejected by the user pool")
    elif not _auth_handler.enabled and SETTINGS.tenant_api_keys:
        # Same trap with tenant keys as the only method: never fall through to
        # the disabled comparison, which accepts anything.
        unauthorized("Credentials rejected: not a tenant API key")
    _auth_handler.verify_credentials(SecretStr(credential) if credential else None)


async def verify_websocket_credentials(
    credential: str | None, scope: Mapping[str, Any] | None = None
) -> None:
    """Verify the credential a WebSocket client presented.

    The HTTP dependency cannot serve this: FastAPI's security schemes are
    annotated ``request: Request`` and the solver fills the *websocket*
    parameter on a WebSocket scope instead, so the scheme is called with no
    argument at all and the handshake fails as a ``TypeError``.

    Args:
        credential: The credential read off the connection, if any.
        scope: The connection's ASGI scope, letting a tenant key's endpoint
            restrictions apply to the matched WebSocket route.

    Raises:
        ApiError: 401 if authentication is required but missing/invalid, or if
            a tenant key is not allowed on this endpoint.
    """
    # Cleared per connection: a session must not inherit another's identity.
    PRINCIPAL.set(None)
    TENANT.set(None)
    await verify_credential(credential)
    if scope is not None:
        enforce_tenant_endpoint_scope(scope)
    elif (tenant := TENANT.get()) is not None:
        # No scope means no matched route to test: fail closed for a
        # restricted tenant rather than skipping its restrictions.
        _enforce_endpoint_scope(tenant, None)


def realtime_signing_key(person: bytes, size: int) -> bytes | None:
    """Return a key derived from this deployment's own API key.

    Args:
        person: Personalisation separating this key from any other derived one.
        size: Length of the derived key, in bytes.

    Returns:
        The derived key, or None when no API key is configured.
    """
    return _auth_handler.derived_key(person, size)
