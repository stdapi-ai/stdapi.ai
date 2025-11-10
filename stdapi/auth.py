"""API key authentication for OpenAI-compatible endpoints."""

from hashlib import blake2b
from hmac import compare_digest
from secrets import token_bytes

from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretBytes, SecretStr
from pydantic_core import from_json

from stdapi.aws import CONFIG, REGION, SESSION
from stdapi.config import SETTINGS
from stdapi.monitoring import log_error_details

#: HTTPBearer security scheme for API key authentication
_security = HTTPBearer(auto_error=False)


class AuthenticationHandler:
    """Handles API key authentication with secure hashing for OpenAI-compatible endpoints."""

    __slots__ = ("_api_key_hash", "_api_key_salt")

    def __init__(self) -> None:
        """Initialize authentication handler with no cached API key hash."""
        self._api_key_hash: SecretBytes | None = None
        self._api_key_salt: SecretBytes | None = None

    def _hash_api_key(self, api_key: SecretStr) -> None:
        """Hash the API key with a random salt using BLAKE2.

        Generates a random 32-byte salt and uses BLAKE2b for secure hashing.
        The salt and hash are stored as instance attributes for later verification.

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

    async def initialize(self) -> bool:
        """Initialize authentication by retrieving and securely hashing the API key.

        This method should be called once during application startup to retrieve
        the API key from the configured source and store its salted hash securely.

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
        if api_key is not None:
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
        async with SESSION.client(
            "ssm", config=CONFIG, region_name=REGION
        ) as ssm_client:
            try:
                return SecretStr(
                    (
                        await ssm_client.get_parameter(
                            Name=SETTINGS.api_key_ssm_parameter, WithDecryption=True
                        )
                    )["Parameter"]["Value"]
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "ParameterNotFound":
                    msg = f"SSM Parameter '{SETTINGS.api_key_ssm_parameter}' not found"
                    raise ValueError(msg) from exc
                raise

    @staticmethod
    async def _get_api_key_from_secrets_manager() -> SecretStr:
        """Retrieve API key from AWS Secrets Manager.

        Returns:
            The API key string from Secrets Manager.

        Raises:
            ClientError: If there's an error retrieving the API key from Secrets Manager.
            ValueError: If the secret or key is not found.
        """
        async with SESSION.client(
            "secretsmanager", config=CONFIG, region_name=REGION
        ) as secrets_client:
            try:
                secret_data = from_json(
                    (
                        await secrets_client.get_secret_value(
                            SecretId=SETTINGS.api_key_secretsmanager_secret
                        )
                    )["SecretString"]
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "ResourceNotFoundException":
                    msg = f"Secret '{SETTINGS.api_key_secretsmanager_secret}' not found"
                    raise ValueError(msg) from exc
                raise
        try:
            return SecretStr(secret_data[SETTINGS.api_key_secretsmanager_key])
        except KeyError as exc:
            msg = (
                f"Key '{SETTINGS.api_key_secretsmanager_key}' not found in secret"
                f" '{SETTINGS.api_key_secretsmanager_secret}'"
            )
            raise ValueError(msg) from exc

    def verify_credentials(
        self, credentials: HTTPAuthorizationCredentials | None
    ) -> None:
        """Verify API key authentication for OpenAI-compatible endpoints.

        This method validates the Authorization header against the cached API key hash
        using secure constant-time comparison. If authentication is disabled, this method
        does nothing and allows all requests.

        Args:
            credentials: HTTP Bearer token credentials from the Authorization header.

        Raises:
            HTTPException: 401 if authentication is required but missing/invalid.
        """
        if self._api_key_hash is None or self._api_key_salt is None:
            return

        if credentials is None:
            log_error_details("Missing API key")
            raise HTTPException(status_code=401, detail="Unauthorized")

        value = SecretStr(credentials.credentials)
        credentials.credentials = ""
        if not compare_digest(
            blake2b(
                value.get_secret_value().encode("utf-8"),
                salt=self._api_key_salt.get_secret_value(),
            ).digest(),
            self._api_key_hash.get_secret_value(),
        ):
            log_error_details("Invalid API key")
            raise HTTPException(status_code=401, detail="Unauthorized")


#: Global authentication handler instance
_auth_handler = AuthenticationHandler()


async def initialize_authentication() -> bool:
    """Initialize the global authentication handler.

    This function should be called once during application startup to retrieve
    and cache the API key from the configured source.

    Returns:
        True if authentication is enabled, False otherwise.
    """
    return await _auth_handler.initialize()


async def authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    """Verify API key authentication dependency for FastAPI routes.

    This dependency validates the Authorization header against the cached API key.
    If authentication is disabled, this dependency does nothing and allows all requests.

    Args:
        credentials: HTTP Bearer token credentials from the Authorization header.

    Raises:
        HTTPException: 401 if authentication is required but missing/invalid.
    """
    _auth_handler.verify_credentials(credentials)
