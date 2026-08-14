"""Amazon Cognito user pool token authentication for API endpoints."""

from asyncio import sleep
from time import monotonic
from typing import TYPE_CHECKING, Any, NoReturn

from aiohttp import ClientError as HttpClientError
from aiohttp import ClientSession, ClientTimeout
from jwt import PyJWKSet, get_unverified_header
from jwt import decode as decode_token
from jwt.exceptions import PyJWTError
from pydantic_core import from_json

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS, cognito_issuer_url
from stdapi.exceptions import ServerError
from stdapi.monitoring import Principal, log_error_details
from stdapi.server import HTTP_CLIENT_HEADERS

if TYPE_CHECKING:
    from jwt import PyJWK
    from jwt.types import Options

#: Signing algorithm Amazon Cognito uses; no other one is accepted.
_ALGORITHMS = ["RS256"]

#: Clock difference tolerated on the time claims of a token, in seconds.
_CLOCK_SKEW = 60

#: Claims every accepted token must carry.
_REQUIRED_CLAIMS = ["iss", "exp", "iat", "sub", "token_use"]

#: Value of the ``token_use`` claim on the token a client presents to call an API.
_USE_ACCESS = "access"

#: Value of the ``token_use`` claim on the token describing the signed-in user.
_USE_ID = "id"

#: Decode options; the application allowlist is checked per token kind instead.
_DECODE_OPTIONS: Options = {"require": _REQUIRED_CLAIMS, "verify_aud": False}

#: Failures of a key set load that leave the cached keys usable.
_KEY_SET_ERRORS = (
    HttpClientError,
    OSError,
    TimeoutError,
    TypeError,
    ValueError,
    PyJWTError,
)

#: Attempts to load the key set at startup before the server refuses to start.
_KEY_SET_ATTEMPTS = 3

#: Delay between two startup key set load attempts, in seconds.
_KEY_SET_RETRY_DELAY = 1.0

#: Timeout of a key set request, in seconds.
_KEY_SET_TIMEOUT = 10.0

#: Largest key set document accepted, in bytes.
_KEY_SET_MAX_SIZE = 65536

#: Delay before an unknown key identifier may trigger another key set load, in seconds.
_KEY_SET_RELOAD_COOLDOWN = 300.0

#: Characters of a token key identifier kept when reporting it in the server log.
_KEY_ID_LOG_LENGTH = 64


def _unauthorized(reason: str) -> NoReturn:
    """Reject the request with the opaque 401 every authentication failure shares.

    Args:
        reason: Why the credential was refused; recorded in the server log only,
            so the response cannot be used to find which check failed.

    Raises:
        ApiError: Always, with status 401.
    """
    log_error_details(reason, status=401)
    msg = "Unauthorized"
    raise ApiError(msg, status=401)


async def _fetch_key_set(url: str) -> Any:  # noqa: ANN401
    """Fetch the JSON Web Key Set a user pool publishes.

    Args:
        url: Key set URL, built once at startup from the configured pool.

    Returns:
        The parsed key set document.

    Raises:
        ValueError: If the document exceeds the accepted size.
    """
    async with (
        ClientSession(
            headers=HTTP_CLIENT_HEADERS, timeout=ClientTimeout(total=_KEY_SET_TIMEOUT)
        ) as session,
        session.get(url) as response,
    ):
        response.raise_for_status()
        document = await response.content.read(_KEY_SET_MAX_SIZE + 1)
    if len(document) > _KEY_SET_MAX_SIZE:
        msg = "The user pool key set document is too large"
        raise ValueError(msg)
    return from_json(document)


class CognitoAuthenticator:
    """Verifies the tokens issued by an Amazon Cognito user pool."""

    __slots__ = (
        "_accept_id_token",
        "_client_ids",
        "_issuer",
        "_key_set_url",
        "_keys",
        "_reload_allowed_at",
        "_required_scopes",
    )

    def __init__(self) -> None:
        """Initialize a disabled authenticator holding no signing key."""
        self._accept_id_token = False
        self._client_ids: frozenset[str] = frozenset()
        self._issuer = ""
        self._key_set_url = ""
        self._keys: dict[str, PyJWK] = {}
        self._reload_allowed_at = 0.0
        self._required_scopes: frozenset[str] = frozenset()

    @property
    def enabled(self) -> bool:
        """Whether user pool authentication is configured.

        Returns:
            True once a user pool is configured and its signing keys are loaded.
        """
        return bool(self._issuer)

    async def initialize(self) -> bool:
        """Load the signing keys of the configured user pool.

        Called once during application startup. Every value derived from the
        configuration is resolved here, so verification stays a cache lookup.

        Returns:
            True if user pool authentication is enabled, False otherwise.

        Raises:
            ServerError: If Amazon Cognito has no endpoint in the pool's Region,
                or the pool's signing keys cannot be loaded, so the server never
                serves requests it could not authenticate.
        """
        user_pool_id = SETTINGS.aws_cognito_user_pool_id
        if not user_pool_id:
            return False
        try:
            issuer = cognito_issuer_url(user_pool_id, SETTINGS.aws_cognito_issuer_type)
        except ValueError as exception:
            raise ServerError(str(exception)) from exception
        self._key_set_url = f"{issuer}/.well-known/jwks.json"
        self._client_ids = frozenset(SETTINGS.aws_cognito_client_ids)
        self._required_scopes = frozenset(SETTINGS.aws_cognito_required_scopes)
        self._accept_id_token = SETTINGS.aws_cognito_accept_id_token
        error: Exception | None = None
        for attempt in range(_KEY_SET_ATTEMPTS):
            if attempt:
                await sleep(_KEY_SET_RETRY_DELAY)
            try:
                await self._load_keys()
            except _KEY_SET_ERRORS as exception:
                error = exception
            else:
                self._issuer = issuer
                return True
        msg = (
            "Unable to load the Amazon Cognito user pool signing keys from "
            f"{self._key_set_url}"
        )
        raise ServerError(msg) from error

    async def verify(self, token: str) -> Principal:
        """Verify a bearer token issued by the configured user pool.

        Args:
            token: Bearer token value sent by the client.

        Returns:
            The caller the token identifies.

        Raises:
            ApiError: 401 if the token is not a valid token for this deployment.
        """
        try:
            header = get_unverified_header(token)
        except PyJWTError:
            _unauthorized("Bearer token is not a signed token")
        key_id = header.get("kid")
        if not isinstance(key_id, str) or not key_id:
            _unauthorized("Bearer token carries no key identifier")
        key = self._keys.get(key_id)
        if key is None:
            await self._reload_keys(key_id)
            key = self._keys.get(key_id)
            if key is None:
                _unauthorized(
                    "Bearer token signed by the unknown key "
                    f"'{key_id[:_KEY_ID_LOG_LENGTH]}'"
                )
        try:
            claims = decode_token(
                token,
                key,
                algorithms=_ALGORITHMS,
                issuer=self._issuer,
                leeway=_CLOCK_SKEW,
                options=_DECODE_OPTIONS,
            )
        except PyJWTError as exception:
            _unauthorized(f"Rejected bearer token: {exception}")
        return self._principal(claims)

    def _principal(self, claims: dict[str, Any]) -> Principal:
        """Authorize a verified claim set and identify its caller.

        Args:
            claims: Claims of a token whose signature, issuer and validity
                period are already verified.

        Returns:
            The caller the claims identify.

        Raises:
            ApiError: 401 if the token is of an unaccepted kind, was issued to
                another application, or lacks a required scope.
        """
        scope = claims.get("scope")
        scopes = frozenset(scope.split()) if isinstance(scope, str) else frozenset()
        token_use = claims.get("token_use")
        if token_use == _USE_ACCESS:
            application = claims.get("client_id")
            username = claims.get("username")
        elif token_use == _USE_ID:
            if not self._accept_id_token:
                _unauthorized("Identity tokens are not accepted by this deployment")
            application = claims.get("aud")
            username = claims.get("cognito:username")
        else:
            _unauthorized("Bearer token is neither an access nor an identity token")
        if not isinstance(application, str) or application not in self._client_ids:
            _unauthorized("Bearer token was issued to another application")
        if not self._required_scopes <= scopes:
            _unauthorized("Bearer token is missing a required scope")
        return Principal(
            subject=claims["sub"],
            username=username if isinstance(username, str) else None,
            client_id=application,
            scopes=scopes,
        )

    async def _load_keys(self) -> None:
        """Fetch the published key set and replace the cached signing keys.

        The cached keys are replaced only once the new set parses, so a pool
        answering with an unusable document does not disable authentication.

        Raises:
            TypeError: If the fetched document is not a key set object.
            ValueError: If the key set holds no key the pool could have signed
                a token with.
        """
        document = await _fetch_key_set(self._key_set_url)
        if not isinstance(document, dict):
            msg = "The user pool key set document is not an object"
            raise TypeError(msg)
        keys = {
            key.key_id: key
            for key in PyJWKSet.from_dict(document).keys
            if key.key_id and key.algorithm_name in _ALGORITHMS
        }
        if not keys:
            msg = "The user pool key set holds no usable signing key"
            raise ValueError(msg)
        self._keys = keys

    async def _reload_keys(self, key_id: str) -> None:
        """Reload the signing keys after an unknown key identifier arrived.

        A pool rotates its keys, so an unknown identifier must be able to
        refresh the cache. It is also chosen by the caller, so it is rate
        limited to one request per cooldown whether that request succeeds or
        not, and a burst of forged identifiers cannot amplify into traffic. The
        cooldown starts before the request is awaited, so callers arriving while
        one reload is in flight return without starting their own -- and without
        waiting for it, so the request that triggered the reload is the first to
        benefit from it.

        Args:
            key_id: Key identifier that is not cached, for the server log.
        """
        if monotonic() < self._reload_allowed_at:
            return
        self._reload_allowed_at = monotonic() + _KEY_SET_RELOAD_COOLDOWN
        try:
            await self._load_keys()
        except _KEY_SET_ERRORS as exception:
            log_error_details(
                f"Unable to reload the Amazon Cognito user pool signing keys "
                f"after a request signed by the unknown key "
                f"'{key_id[:_KEY_ID_LOG_LENGTH]}': {exception}"
            )
