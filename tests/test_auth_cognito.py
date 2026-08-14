"""Amazon Cognito user pool token verification in :mod:`stdapi.auth_cognito`.

Every claim shape asserted here is the one a real user pool issues: an access
token carries ``client_id`` and no ``aud``, an identity token carries ``aud``
and no ``client_id``, neither carries ``nbf``, and a pool signs the two token
kinds with two different keys. The rejection cases are the attacks the
verification exists for -- algorithm confusion, an unknown key id used as a
fetch amplifier, another pool's issuer, and a token minted for another
application.

Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
     stdapi/auth_cognito.py:CognitoAuthenticator
"""

from __future__ import annotations

from asyncio import Event, gather, sleep
from hashlib import sha256, sha384, sha512
from hmac import new as hmac_new
from json import dumps
from time import time
from typing import TYPE_CHECKING, Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.security import HTTPAuthorizationCredentials
from jwt import encode as jwt_encode
from jwt.utils import base64url_encode, to_base64url_uint
from pydantic import SecretStr, ValidationError
from starlette.testclient import TestClient

import stdapi.auth
import stdapi.auth_cognito
from stdapi.api_errors import ApiError
from stdapi.auth import AuthenticationHandler, authenticate, initialize_authentication
from stdapi.auth_cognito import CognitoAuthenticator
from stdapi.config import SETTINGS, _Settings
from stdapi.exceptions import ServerError
from stdapi.monitoring import PRINCIPAL, Principal
from stdapi.routes import core_models
from tests._helpers import make_event_log

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

pytestmark = pytest.mark.local

#: Identifier of the user pool the tokens below are minted for.
POOL_ID = "eu-west-3_hAsOX9Vdf"

#: Issuer a pool with the original issuer configuration puts in every token.
ISSUER = f"https://cognito-idp.eu-west-3.amazonaws.com/{POOL_ID}"

#: Issuer a pool with the updated issuer configuration puts in every token.
UPDATED_ISSUER = f"https://issuer-cognito-idp.eu-west-3.amazonaws.com/{POOL_ID}"

#: Application the tokens below are minted for.
CLIENT_ID = "158vqcm5r8ffkgqbabcdefghij"

#: Key identifier the pool signs access tokens with.
ACCESS_KID = "ONFiIkzeALpemXRGSlOsSPhwZr8XvHjZa1FOQTP7kJM="

#: Key identifier the pool signs identity tokens with.
ID_KID = "k2UdSLM8O6alapUB4JvVnTjI9o/BA9w1G+ryxKksDC0="

#: Only scope a user-password sign-in yields; no resource server is involved.
ADMIN_SCOPE = "aws.cognito.signin.user.admin"


class Harness(NamedTuple):
    """An initialized authenticator and the key-set URLs it has fetched."""

    authenticator: CognitoAuthenticator
    fetches: list[str]


@pytest.fixture(scope="session")
def signing_keys() -> dict[str, RSAPrivateKey]:
    """RSA keypairs standing in for the two keys a user pool publishes.

    A pool signs access tokens and identity tokens with different keys, so a
    single-key cache would pass every test while failing on the first real
    identity token.
    """
    return {
        kid: generate_private_key(public_exponent=65537, key_size=2048)
        for kid in (ACCESS_KID, ID_KID)
    }


@pytest.fixture(scope="session")
def foreign_key() -> RSAPrivateKey:
    """A keypair the pool never published, for the wrong-signer case."""
    return generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def jwks_document(signing_keys: dict[str, RSAPrivateKey]) -> dict[str, Any]:
    """The JSON Web Key Set document the pool serves for those keys."""
    return {
        "keys": [
            {
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "kid": kid,
                "n": to_base64url_uint(key.public_key().public_numbers().n).decode(),
                "e": to_base64url_uint(key.public_key().public_numbers().e).decode(),
            }
            for kid, key in signing_keys.items()
        ]
    }


def access_claims(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the claim set a real access token carries, with *overrides* applied.

    Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-with-identity-providers.html
    """
    now = int(time())
    return {
        "sub": "b7d3f0a2-1c4e-4f6a-9f1b-2d5e8c0a3b4c",
        "iss": ISSUER,
        "client_id": CLIENT_ID,
        "origin_jti": "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
        "event_id": "2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e",
        "token_use": "access",
        "scope": ADMIN_SCOPE,
        "auth_time": now,
        "exp": now + 3600,
        "iat": now,
        "jti": "3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f",
        "username": "alice",
        **overrides,
    }


def id_claims(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the claim set a real identity token carries, with *overrides* applied."""
    now = int(time())
    return {
        "sub": "b7d3f0a2-1c4e-4f6a-9f1b-2d5e8c0a3b4c",
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "cognito:username": "alice",
        "email": "alice@example.com",
        "token_use": "id",
        "auth_time": now,
        "exp": now + 3600,
        "iat": now,
        "jti": "4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f90",
        **overrides,
    }


def mint(
    claims: dict[str, Any],
    key: RSAPrivateKey,
    *,
    kid: str = ACCESS_KID,
    algorithm: str = "RS256",
) -> str:
    """Sign *claims* the way the pool does."""
    return jwt_encode(claims, key, algorithm=algorithm, headers={"kid": kid})


def forge(claims: dict[str, Any], header: dict[str, Any], secret: bytes = b"") -> str:
    """Hand-build a token whose header is *header*, signed with an HMAC over *secret*.

    ``jwt.encode`` refuses to HMAC with an asymmetric key, so the algorithm
    confusion attack -- signing with the public key as the shared secret -- can
    only be built segment by segment.
    """
    segments = [
        base64url_encode(dumps(header, separators=(",", ":")).encode()),
        base64url_encode(dumps(claims, separators=(",", ":")).encode()),
    ]
    signing_input = b".".join(segments)
    digest = {"HS256": sha256, "HS384": sha384, "HS512": sha512}.get(header["alg"])
    signature = (
        hmac_new(secret, signing_input, digest).digest() if digest is not None else b""
    )
    return b".".join((*segments, base64url_encode(signature))).decode()


def public_pem(key: RSAPrivateKey) -> bytes:
    """Return the PEM-encoded public key, the usual algorithm-confusion secret."""
    return key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )


@pytest.fixture
def make_authenticator(
    monkeypatch: pytest.MonkeyPatch, jwks_document: dict[str, Any]
) -> Callable[..., Coroutine[Any, Any, Harness]]:
    """Return a factory building an initialized authenticator with a stubbed key fetch.

    Only the HTTPS call is replaced: the key set is parsed, cached and selected
    by the code under test, and every fetch is recorded so the refresh rate
    limit can be asserted.
    """

    async def _make(**settings: Any) -> Harness:  # noqa: ANN401
        fetches: list[str] = []

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            fetches.append(url)
            return jwks_document

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", POOL_ID)
        monkeypatch.setattr(SETTINGS, "aws_cognito_client_ids", [CLIENT_ID])
        monkeypatch.setattr(SETTINGS, "aws_cognito_required_scopes", [])
        monkeypatch.setattr(SETTINGS, "aws_cognito_accept_id_token", False)
        monkeypatch.setattr(SETTINGS, "aws_cognito_issuer_type", "original")
        for name, value in settings.items():
            monkeypatch.setattr(SETTINGS, name, value)
        authenticator = CognitoAuthenticator()
        assert await authenticator.initialize() is True
        return Harness(authenticator, fetches)

    return _make


@pytest.mark.usefixtures("request_log")
class TestAcceptedTokens:
    """What a valid pool token yields, and the tolerances applied to its clock claims.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
    """

    async def test_access_token_is_accepted_and_identifies_the_caller(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A real access token yields the caller's subject, name, application and scopes.

        The claim set carries no ``aud`` and no ``nbf``, which is what a pool
        actually issues; requiring either would reject every real token.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = mint(access_claims(), signing_keys[ACCESS_KID])

        principal = await harness.authenticator.verify(token)

        assert principal.subject == "b7d3f0a2-1c4e-4f6a-9f1b-2d5e8c0a3b4c"
        assert principal.username == "alice"
        assert principal.client_id == CLIENT_ID
        assert principal.scopes == frozenset({ADMIN_SCOPE})

    async def test_happy_path_never_fetches_the_key_set(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """Verification is O(1): the key set is fetched once, at startup, and reused.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = mint(access_claims(), signing_keys[ACCESS_KID])

        for _ in range(5):
            await harness.authenticator.verify(token)

        assert len(harness.fetches) == 1

    async def test_machine_to_machine_token_has_no_username(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A client-credentials token identifies an application, not a person.

        Its ``sub`` is the application id and it carries no ``username``, so the
        principal must not assume a human caller.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator(
            aws_cognito_required_scopes=["stdapi/invoke"]
        )
        claims = access_claims(scope="stdapi/invoke", sub=CLIENT_ID, version=2)
        del claims["username"]
        token = mint(claims, signing_keys[ACCESS_KID])

        principal = await harness.authenticator.verify(token)

        assert principal.subject == CLIENT_ID
        assert principal.username is None
        assert principal.scopes == frozenset({"stdapi/invoke"})

    async def test_token_expired_inside_the_clock_skew_is_accepted(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A token just past ``exp`` still verifies, so a small clock drift is not an outage.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        now = int(time())
        token = mint(
            access_claims(exp=now - 30, iat=now - 300), signing_keys[ACCESS_KID]
        )

        assert await harness.authenticator.verify(token)

    async def test_identity_token_is_accepted_when_opted_in(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """An identity token names its application in ``aud`` and its user in ``cognito:username``.

        It is signed with the pool's other key, so accepting it also proves both
        published keys are cached.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator(aws_cognito_accept_id_token=True)
        token = mint(id_claims(), signing_keys[ID_KID], kid=ID_KID)

        principal = await harness.authenticator.verify(token)

        assert principal.username == "alice"
        assert principal.client_id == CLIENT_ID
        assert principal.scopes == frozenset()

    async def test_updated_issuer_pool_is_accepted_when_configured(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A pool with the updated issuer configuration signs a different ``iss``.

        Ref: https://docs.aws.amazon.com/cognito-user-identity-pools/latest/APIReference/API_IssuerConfigurationType.html
        """
        harness = await make_authenticator(aws_cognito_issuer_type="updated")
        token = mint(access_claims(iss=UPDATED_ISSUER), signing_keys[ACCESS_KID])

        assert await harness.authenticator.verify(token)
        assert harness.fetches == [f"{UPDATED_ISSUER}/.well-known/jwks.json"]


@pytest.mark.usefixtures("request_log")
class TestRejectedSignatures:
    """Everything that tries to be signed by the pool and is not.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
    """

    @staticmethod
    def _assert_unauthorized(excinfo: pytest.ExceptionInfo[ApiError]) -> None:
        """Assert the failure is the opaque 401 every rejection shares."""
        assert excinfo.value.status == 401
        assert str(excinfo.value) == "Unauthorized"

    async def test_unsigned_token_is_rejected(
        self, make_authenticator: Callable[..., Coroutine[Any, Any, Harness]]
    ) -> None:
        """``alg=none`` removes the signature entirely and must never be honoured.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = forge(access_claims(), {"alg": "none", "kid": ACCESS_KID, "typ": "JWT"})

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        self._assert_unauthorized(excinfo)

    @pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
    async def test_public_key_as_hmac_secret_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        algorithm: str,
    ) -> None:
        """The classic algorithm confusion attack: HMAC the token with the public key.

        The public key is published in the key set, so a verifier that trusts
        the token's own ``alg`` would accept a token anyone can mint.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = forge(
            access_claims(),
            {"alg": algorithm, "kid": ACCESS_KID, "typ": "JWT"},
            public_pem(signing_keys[ACCESS_KID]),
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        self._assert_unauthorized(excinfo)

    async def test_key_set_modulus_as_hmac_secret_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        jwks_document: dict[str, Any],
    ) -> None:
        """The same attack using the raw modulus published in the key set.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        modulus = next(
            key["n"] for key in jwks_document["keys"] if key["kid"] == ACCESS_KID
        )
        token = forge(
            access_claims(),
            {"alg": "HS256", "kid": ACCESS_KID, "typ": "JWT"},
            modulus.encode(),
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        self._assert_unauthorized(excinfo)

    async def test_empty_hmac_secret_is_rejected(
        self, make_authenticator: Callable[..., Coroutine[Any, Any, Harness]]
    ) -> None:
        """An HMAC token signed with an empty secret is rejected like any other.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = forge(
            access_claims(), {"alg": "HS256", "kid": ACCESS_KID, "typ": "JWT"}
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        self._assert_unauthorized(excinfo)

    async def test_stronger_rsa_algorithm_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """Only RS256 is accepted: a pool signs with nothing else.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
        """
        harness = await make_authenticator()
        token = mint(access_claims(), signing_keys[ACCESS_KID], algorithm="RS512")

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        self._assert_unauthorized(excinfo)

    async def test_token_signed_by_another_key_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        foreign_key: RSAPrivateKey,
    ) -> None:
        """A well-formed token signed by a key the pool never published is rejected.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = mint(access_claims(), foreign_key)

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        self._assert_unauthorized(excinfo)

    async def test_the_reason_is_written_to_the_server_log_only(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        foreign_key: RSAPrivateKey,
        request_log: dict[str, Any],
    ) -> None:
        """The client is told nothing; the operator is told which check failed.

        Ref: stdapi/monitoring.py:log_error_details
        """
        harness = await make_authenticator()

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(mint(access_claims(), foreign_key))

        assert str(excinfo.value) == "Unauthorized"
        assert any(
            "signature" in str(detail).lower() for detail in request_log["error_detail"]
        )

    async def test_a_refused_credential_is_a_warning_not_an_incident(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        foreign_key: RSAPrivateKey,
        request_log: dict[str, Any],
    ) -> None:
        """The entry keeps the severity an ordinary 401 deserves.

        An expired browser token is a daily event; logged without its status it
        resolves to ``critical`` and every one of them reads as an outage.

        Ref: stdapi/auth_cognito.py:_unauthorized
             stdapi/monitoring.py:_error_level
        """
        harness = await make_authenticator()

        with pytest.raises(ApiError):
            await harness.authenticator.verify(mint(access_claims(), foreign_key))

        assert request_log["level"] == "warning"


@pytest.mark.usefixtures("request_log")
class TestRejectedClaims:
    """Tokens the pool could have signed, for someone else or for something else.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
    """

    async def test_another_pools_issuer_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A token from another pool in the same account is not a token for this one.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = mint(
            access_claims(
                iss="https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_other"
            ),
            signing_keys[ACCESS_KID],
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_updated_issuer_is_rejected_when_the_original_is_configured(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """The two issuer forms are not interchangeable: only the configured one is valid.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = mint(access_claims(iss=UPDATED_ISSUER), signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_original_issuer_is_rejected_when_the_updated_is_configured(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """The mirror case, so neither form is accepted merely because both are known.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator(aws_cognito_issuer_type="updated")
        token = mint(access_claims(), signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_unknown_application_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """An access token names its application in ``client_id``, which must be allowed.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator()
        token = mint(
            access_claims(client_id="other-application"), signing_keys[ACCESS_KID]
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_identity_token_is_rejected_by_default(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """Identity tokens are not access tokens and are refused unless opted in.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator()
        token = mint(id_claims(), signing_keys[ID_KID], kid=ID_KID)

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_identity_token_for_another_application_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """With identity tokens enabled, ``aud`` is checked against the same allowlist.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator(aws_cognito_accept_id_token=True)
        token = mint(
            id_claims(aud="other-application"), signing_keys[ID_KID], kid=ID_KID
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_refresh_token_use_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """Only the token kinds the gateway understands are accepted.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator()
        claims = access_claims(token_use="refresh")  # noqa: S106
        token = mint(claims, signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_expired_token_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """Past the skew, an expired token is refused.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        now = int(time())
        token = mint(
            access_claims(exp=now - 600, iat=now - 4200), signing_keys[ACCESS_KID]
        )

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_token_not_yet_valid_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """``nbf`` is honoured when present, even though no pool token carries one.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = mint(access_claims(nbf=int(time()) + 600), signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    @pytest.mark.parametrize("claim", ["sub", "exp", "iat", "token_use", "iss"])
    async def test_missing_required_claim_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        claim: str,
    ) -> None:
        """Each mandatory claim is mandatory on its own.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        claims = access_claims()
        del claims[claim]
        token = mint(claims, signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_access_token_without_client_id_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """An access token that names no application cannot be matched to the allowlist.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator()
        claims = access_claims()
        del claims["client_id"]
        token = mint(claims, signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_missing_required_scope_is_rejected(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A sign-in token carries only the built-in scope, not a resource server one.

        This is the failure an operator meets when they require ``stdapi/invoke``
        and their clients sign in with a username and password.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator(
            aws_cognito_required_scopes=["stdapi/invoke"]
        )
        token = mint(access_claims(), signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401

    async def test_all_required_scopes_must_be_present(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """Holding one of two required scopes is not enough.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        harness = await make_authenticator(
            aws_cognito_required_scopes=["stdapi/invoke", "stdapi/admin"]
        )
        token = mint(access_claims(scope="stdapi/invoke"), signing_keys[ACCESS_KID])

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401


@pytest.mark.usefixtures("request_log")
class TestMalformedTokens:
    """Structurally invalid input is refused before anything is fetched or parsed.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
    """

    @pytest.mark.parametrize(
        ("label", "token"),
        [
            ("empty", ""),
            ("not a token", "a.b.c"),
            ("two segments", "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0"),
            ("garbage", "!" * 200_000),
        ],
    )
    async def test_malformed_token_is_rejected_without_a_fetch(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        label: str,
        token: str,
    ) -> None:
        """Nothing that is not a signed token reaches the key set.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401, label
        assert len(harness.fetches) == 1, label

    async def test_non_object_payload_is_rejected(
        self, make_authenticator: Callable[..., Coroutine[Any, Any, Harness]]
    ) -> None:
        """A syntactically valid token whose payload is not a claim set is refused.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        segments = [
            base64url_encode(dumps({"alg": "RS256", "kid": ACCESS_KID}).encode()),
            base64url_encode(b'"not-an-object"'),
            base64url_encode(b"signature"),
        ]

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(b".".join(segments).decode())
        assert excinfo.value.status == 401

    async def test_non_string_key_id_is_rejected_without_a_fetch(
        self, make_authenticator: Callable[..., Coroutine[Any, Any, Harness]]
    ) -> None:
        """A ``kid`` that is not a string cannot address a cached key.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = forge(access_claims(), {"alg": "RS256", "kid": 12345})

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401
        assert len(harness.fetches) == 1

    async def test_token_without_a_key_id_is_rejected_without_a_fetch(
        self, make_authenticator: Callable[..., Coroutine[Any, Any, Harness]]
    ) -> None:
        """A token naming no key cannot be matched against the published set.

        Every pool token names its signing key, so a token without one is not
        an occasion to reload the keys.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = forge(access_claims(), {"alg": "RS256"})

        with pytest.raises(ApiError) as excinfo:
            await harness.authenticator.verify(token)
        assert excinfo.value.status == 401
        assert len(harness.fetches) == 1

    async def test_oversized_key_id_is_truncated_before_it_is_logged(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        request_log: dict[str, Any],
    ) -> None:
        """A caller-chosen ``kid`` is unbounded, so it is capped before reaching the log.

        Left whole it would let any client write a 100 kB line into the request
        log on every rejected request.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
        """
        harness = await make_authenticator()
        token = forge(access_claims(), {"alg": "RS256", "kid": "x" * 100_000})

        with pytest.raises(ApiError):
            await harness.authenticator.verify(token)

        assert max(len(str(detail)) for detail in request_log["error_detail"]) < 200


@pytest.mark.usefixtures("request_log")
class TestKeySetRefresh:
    """An unknown key id refreshes the cached keys, at most once per cooldown.

    A pool rotates its keys, so an unknown key id must be able to refresh the
    cache; a forged one must not turn every request into an outbound fetch.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
    """

    async def test_unknown_key_id_triggers_exactly_one_refresh(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """The first unknown key id refreshes; a burst of forged ones does not.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
        """
        harness = await make_authenticator()

        for index in range(20):
            token = mint(
                access_claims(), signing_keys[ACCESS_KID], kid=f"rotated-{index}"
            )
            with pytest.raises(ApiError):
                await harness.authenticator.verify(token)

        assert len(harness.fetches) == 2

    async def test_rotated_key_is_picked_up_by_the_refresh(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        jwks_document: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A key added to the published set after startup verifies its tokens.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
        """
        harness = await make_authenticator()
        rotated = dict(jwks_document["keys"][0], kid="rotated-key")

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            harness.fetches.append(url)
            return {"keys": [*jwks_document["keys"], rotated]}

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        token = mint(access_claims(), signing_keys[ACCESS_KID], kid="rotated-key")

        assert await harness.authenticator.verify(token)
        assert len(harness.fetches) == 2

    async def test_a_failing_refresh_does_not_lift_the_cooldown(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A refresh that errors still counts, so a failing endpoint is not hammered.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
        """
        harness = await make_authenticator()

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            harness.fetches.append(url)
            msg = "unreachable"
            raise TimeoutError(msg)

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)

        for _ in range(5):
            token = mint(access_claims(), signing_keys[ACCESS_KID], kid="rotated-key")
            with pytest.raises(ApiError) as excinfo:
                await harness.authenticator.verify(token)
            assert excinfo.value.status == 401

        assert len(harness.fetches) == 2

    async def test_the_cooldown_expires_and_allows_the_next_refresh(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once the window has passed, an unknown key id refreshes again.

        A cooldown that never lifted would turn one forged key id into a full
        authentication outage: the pool rotates its keys and every genuine token
        is rejected until the process restarts.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
        """
        harness = await make_authenticator()
        monkeypatch.setattr(stdapi.auth_cognito, "_KEY_SET_RELOAD_COOLDOWN", 0.0)

        for index in range(2):
            token = mint(
                access_claims(), signing_keys[ACCESS_KID], kid=f"rotated-{index}"
            )
            with pytest.raises(ApiError):
                await harness.authenticator.verify(token)

        assert len(harness.fetches) == 3

    async def test_keys_survive_a_key_set_that_stops_parsing(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A refresh answering with an unusable document leaves the cached keys in place.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
        """
        harness = await make_authenticator()

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            harness.fetches.append(url)
            return {"keys": []}

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        with pytest.raises(ApiError):
            await harness.authenticator.verify(
                mint(access_claims(), signing_keys[ACCESS_KID], kid="rotated-key")
            )

        assert await harness.authenticator.verify(
            mint(access_claims(), signing_keys[ACCESS_KID])
        )

    async def test_keys_survive_a_key_set_that_is_not_a_document(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An endpoint answering with something that is not a key set changes nothing.

        A captive portal or a proxy error page is the realistic source of a
        reply that parses as JSON but is not an object.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator._load_keys
        """
        harness = await make_authenticator()

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            harness.fetches.append(url)
            return ["not", "a", "key", "set"]

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        with pytest.raises(ApiError):
            await harness.authenticator.verify(
                mint(access_claims(), signing_keys[ACCESS_KID], kid="rotated-key")
            )

        assert await harness.authenticator.verify(
            mint(access_claims(), signing_keys[ACCESS_KID])
        )

    async def test_concurrent_unknown_key_ids_trigger_a_single_reload(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        jwks_document: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Requests racing on the same unknown key reload the keys once, not once each.

        Without single-flight, a burst arriving together would each pass the
        cooldown check before any of them had recorded a reload.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator._reload_keys
        """
        harness = await make_authenticator()
        started = Event()

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            harness.fetches.append(url)
            started.set()
            await sleep(0)
            return jwks_document

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        token = mint(access_claims(), signing_keys[ACCESS_KID], kid="rotated-key")

        results = await gather(
            *(harness.authenticator.verify(token) for _ in range(5)),
            return_exceptions=True,
        )

        assert started.is_set()
        assert all(isinstance(result, ApiError) for result in results)
        assert len(harness.fetches) == 2

    async def test_a_key_set_without_an_accepted_algorithm_is_refused(
        self,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
        jwks_document: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keys published for another algorithm are not usable and are not cached.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._load_keys
        """
        harness = await make_authenticator()

        async def _fetch(url: str) -> Any:  # noqa: ANN401
            harness.fetches.append(url)
            return {"keys": [dict(jwks_document["keys"][0], alg="RS512")]}

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        with pytest.raises(ApiError):
            await harness.authenticator.verify(
                mint(access_claims(), signing_keys[ACCESS_KID], kid="rotated-key")
            )

        assert await harness.authenticator.verify(
            mint(access_claims(), signing_keys[ACCESS_KID])
        )


class TestKeySetRequest:
    """The HTTPS request that reads the pool's published keys.

    Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
         stdapi/auth_cognito.py:_fetch_key_set
    """

    @staticmethod
    def _session(body: bytes) -> MagicMock:
        """Stand in for an ``aiohttp`` session answering *body* to a GET."""
        response = AsyncMock()
        response.__aenter__ = AsyncMock(return_value=response)
        response.content.read = AsyncMock(return_value=body)
        response.raise_for_status = MagicMock()
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.get = MagicMock(return_value=response)
        return session

    async def test_document_is_read_from_the_requested_url(
        self, monkeypatch: pytest.MonkeyPatch, jwks_document: dict[str, Any]
    ) -> None:
        """The key set is read from the URL it is asked for, and parsed as JSON.

        Ref: stdapi/auth_cognito.py:_fetch_key_set
        """
        session = self._session(dumps(jwks_document).encode())
        monkeypatch.setattr(
            stdapi.auth_cognito, "ClientSession", MagicMock(return_value=session)
        )
        url = f"{ISSUER}/.well-known/jwks.json"

        document = await stdapi.auth_cognito._fetch_key_set(url)  # noqa: SLF001

        assert session.get.call_args.args == (url,)
        assert document == jwks_document

    async def test_oversized_document_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key set larger than the accepted size is refused instead of parsed.

        The endpoint is reached over the network, so its answer is bounded
        rather than trusted to be the small document a pool publishes.

        Ref: stdapi/auth_cognito.py:_fetch_key_set
        """
        oversized = b"x" * (stdapi.auth_cognito._KEY_SET_MAX_SIZE + 1)  # noqa: SLF001
        monkeypatch.setattr(
            stdapi.auth_cognito,
            "ClientSession",
            MagicMock(return_value=self._session(oversized)),
        )

        with pytest.raises(ValueError, match="too large"):
            await stdapi.auth_cognito._fetch_key_set(ISSUER)  # noqa: SLF001


class TestStartup:
    """The gateway refuses to serve rather than start without the pool's keys.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator.initialize
    """

    async def test_unreachable_key_set_fails_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Booting without keys would reject every token; startup fails instead.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.initialize
        """
        attempts = 0

        async def _fetch(url: str) -> Any:  # noqa: ANN401, ARG001
            nonlocal attempts
            attempts += 1
            msg = "unreachable"
            raise TimeoutError(msg)

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        monkeypatch.setattr(stdapi.auth_cognito, "_KEY_SET_RETRY_DELAY", 0.0)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", POOL_ID)
        monkeypatch.setattr(SETTINGS, "aws_cognito_client_ids", [CLIENT_ID])

        with pytest.raises(ServerError):
            await CognitoAuthenticator().initialize()
        assert attempts > 1, "the fetch must be retried before startup is abandoned"

    async def test_unconfigured_pool_leaves_the_authenticator_disabled(self) -> None:
        """Without a pool id the feature is off and nothing is fetched.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator.initialize
        """
        assert await CognitoAuthenticator().initialize() is False

    async def test_configured_pool_silences_the_no_authentication_warning(
        self, monkeypatch: pytest.MonkeyPatch, jwks_document: dict[str, Any]
    ) -> None:
        """A pool alone is authentication, so the start event must not warn.

        Ref: stdapi/auth.py:initialize_authentication
        """

        async def _fetch(url: str) -> Any:  # noqa: ANN401, ARG001
            return jwks_document

        monkeypatch.setattr(stdapi.auth_cognito, "_fetch_key_set", _fetch)
        monkeypatch.setattr(SETTINGS, "api_key", None)
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", POOL_ID)
        monkeypatch.setattr(SETTINGS, "aws_cognito_client_ids", [CLIENT_ID])
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", CognitoAuthenticator()
        )
        start_event = make_event_log(type="start")

        await initialize_authentication(start_event)

        assert "server_warnings" not in start_event

    async def test_an_empty_api_key_source_fails_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key source holding nothing is a misconfiguration, never an open gateway.

        A blanked or not-yet-populated secret passes configuration validation --
        the source is named -- so only startup can catch it.

        Ref: stdapi/auth.py:initialize_authentication
        """

        async def _empty_key() -> SecretStr:
            return SecretStr("")

        monkeypatch.setattr(
            AuthenticationHandler, "_get_api_key_from_ssm", staticmethod(_empty_key)
        )
        monkeypatch.setattr(SETTINGS, "api_key", None)
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", "/stdapi/api-key")
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", None)
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", CognitoAuthenticator()
        )

        with pytest.raises(ServerError):
            await initialize_authentication(make_event_log(type="start"))

    async def test_a_demanded_method_that_is_not_enabled_fails_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mode asserts a posture, so startup fails rather than degrade to none.

        Ref: stdapi/auth.py:initialize_authentication
        """
        monkeypatch.setattr(SETTINGS, "authentication_mode", "api_key")
        monkeypatch.setattr(SETTINGS, "api_key", None)
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", None)
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", CognitoAuthenticator()
        )

        with pytest.raises(ServerError):
            await initialize_authentication(make_event_log(type="start"))

    async def test_no_method_configured_at_all_only_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running without authentication stays a supported, warned-about choice.

        Ref: stdapi/auth.py:initialize_authentication
        """
        monkeypatch.setattr(SETTINGS, "authentication_mode", "any")
        monkeypatch.setattr(SETTINGS, "api_key", None)
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        monkeypatch.setattr(SETTINGS, "aws_cognito_user_pool_id", None)
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", CognitoAuthenticator()
        )
        start_event = make_event_log(type="start")

        await initialize_authentication(start_event)

        assert start_event["server_warnings"]


class TestConfigurationValidation:
    """A half-configured pool fails startup instead of silently disabling authentication.

    Ref: stdapi/config.py:_Settings._validate_cognito
    """

    @staticmethod
    def _settings(**overrides: Any) -> _Settings:  # noqa: ANN401
        """Build a settings object with the Cognito fields under test.

        The API key sources and the OAuth discovery settings are unset unless a
        case sets them: the suite's own environment carries both, which would
        otherwise decide the outcome of every ``authentication_mode`` case and
        publish an issuer naming a pool no case configures.
        """
        return _Settings(
            aws_bedrock_regions=["us-east-1"],
            **{
                "api_key": None,
                "api_key_ssm_parameter": None,
                "api_key_secretsmanager_secret": None,
                "oauth_resource_identifier": None,
                "oauth_authorization_servers": [],
                "oauth_scopes_supported": [],
                **overrides,
            },
        )

    def test_pool_without_an_application_allowlist_is_rejected(self) -> None:
        """An empty allowlist would accept tokens minted for any application.

        Ref: stdapi/config.py:_Settings._validate_cognito
        """
        with pytest.raises(ValidationError, match="aws_cognito_client_ids"):
            self._settings(aws_cognito_user_pool_id=POOL_ID)

    def test_application_allowlist_without_a_pool_is_rejected(self) -> None:
        """Configuration that cannot take effect is an error, never a silent no-op.

        Ref: stdapi/config.py:_Settings._validate_cognito
        """
        with pytest.raises(ValidationError, match="aws_cognito_user_pool_id"):
            self._settings(aws_cognito_client_ids=[CLIENT_ID])

    def test_malformed_pool_id_is_rejected(self) -> None:
        """The pool id carries its own Region, so a malformed one has no issuer.

        Ref: stdapi/config.py:_Settings._validate_cognito
        """
        with pytest.raises(ValidationError, match="aws_cognito_user_pool_id"):
            self._settings(
                aws_cognito_user_pool_id="not-a-pool",
                aws_cognito_client_ids=[CLIENT_ID],
            )

    def test_required_authentication_method_must_be_configured(self) -> None:
        """Requiring a method nobody configured would leave the gateway open.

        Ref: stdapi/config.py:_Settings._validate_authentication_mode
        """
        with pytest.raises(ValidationError, match="authentication_mode"):
            self._settings(authentication_mode="cognito")

    def test_api_key_only_mode_rejects_a_configured_pool(self) -> None:
        """A configured pool that the mode ignores is a misconfiguration, not a default.

        Ref: stdapi/config.py:_Settings._validate_authentication_mode
        """
        with pytest.raises(ValidationError, match="authentication_mode"):
            self._settings(
                authentication_mode="api_key",
                api_key="a-key",
                aws_cognito_user_pool_id=POOL_ID,
                aws_cognito_client_ids=[CLIENT_ID],
            )

    @pytest.mark.parametrize(
        "source",
        [
            {"api_key": "a-key"},
            {"api_key_ssm_parameter": "/stdapi/api-key"},
            {"api_key_secretsmanager_secret": "prod/stdapi/api-key"},
        ],
    )
    def test_pool_only_mode_rejects_a_configured_api_key(
        self, source: dict[str, str]
    ) -> None:
        """The mirror case: an API key nobody may use is refused too.

        Every source counts, not just the inline value: an indirect one arms the
        same handler at startup, so a pool-only deployment would still accept a
        key it declared it would not.

        Ref: stdapi/config.py:_Settings._validate_authentication_mode
        """
        with pytest.raises(ValidationError, match="authentication_mode"):
            self._settings(
                authentication_mode="cognito",
                aws_cognito_user_pool_id=POOL_ID,
                aws_cognito_client_ids=[CLIENT_ID],
                **source,
            )

    def test_api_key_mode_without_a_key_source_is_rejected(self) -> None:
        """The mode's own case: demanding the API key without configuring one.

        Nothing else catches it, and the deployment would start with no
        authentication method enabled at all.

        Ref: stdapi/config.py:_Settings._validate_authentication_mode
        """
        with pytest.raises(ValidationError, match="authentication_mode"):
            self._settings(authentication_mode="api_key")

    def test_issuer_type_without_a_pool_is_rejected(self) -> None:
        """A pool setting that cannot take effect fails startup like the others.

        Ref: stdapi/config.py:_Settings._validate_cognito
        """
        with pytest.raises(ValidationError, match="aws_cognito_user_pool_id"):
            self._settings(aws_cognito_issuer_type="updated")

    def test_both_methods_configured_is_accepted(self) -> None:
        """The default mode accepts every method that is configured.

        Ref: stdapi/config.py:_Settings._validate_authentication_mode
        """
        settings = self._settings(
            api_key="a-key",
            aws_cognito_user_pool_id=POOL_ID,
            aws_cognito_client_ids=[CLIENT_ID],
        )
        assert settings.authentication_mode == "any"

    def test_application_allowlist_accepts_a_comma_separated_list(self) -> None:
        """Container environments carry lists as comma-separated strings.

        Ref: stdapi/config.py:_Settings._parse_comma_list
        """
        settings = self._settings(
            aws_cognito_user_pool_id=POOL_ID,
            aws_cognito_client_ids=f"{CLIENT_ID}, other-client,",
        )
        assert settings.aws_cognito_client_ids == [CLIENT_ID, "other-client"]


@pytest.mark.usefixtures("request_log")
class TestCredentialDispatch:
    """Which credential the gateway validates when both methods are configured.

    Ref: stdapi/auth.py:authenticate
    """

    @staticmethod
    async def _enable_both(
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> Harness:
        """Install a real API-key handler and a real pool authenticator."""
        harness = await make_authenticator()
        handler = AuthenticationHandler()
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("good-key"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        assert await handler.initialize() is True
        monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", harness.authenticator
        )
        return harness

    async def test_pool_token_is_accepted_and_binds_the_principal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A valid token authenticates the request and identifies its caller.

        Ref: stdapi/auth.py:authenticate
        """
        await self._enable_both(monkeypatch, make_authenticator)
        token = mint(access_claims(), signing_keys[ACCESS_KID])

        await authenticate(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=token
            ),
            x_api_key=None,
        )

        principal = PRINCIPAL.get()
        assert principal is not None
        assert principal.subject == "b7d3f0a2-1c4e-4f6a-9f1b-2d5e8c0a3b4c"

    async def test_api_key_is_still_accepted_and_carries_no_principal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> None:
        """The API key keeps working unchanged, and identifies no person.

        Ref: stdapi/auth.py:authenticate
        """
        await self._enable_both(monkeypatch, make_authenticator)

        await authenticate(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="good-key"
            ),
            x_api_key=None,
        )
        assert PRINCIPAL.get() is None

        await authenticate(credentials=None, x_api_key="good-key")
        assert PRINCIPAL.get() is None

    async def test_a_principal_never_survives_into_the_next_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> None:
        """A request authenticated by the API key clears whatever ran before it.

        A tool call reaches the API as a second request running inside the
        first one's context, so a principal left in place would attribute the
        nested request -- its usage, and the identity it runs under -- to the
        outer caller it never authenticated.

        Ref: stdapi/auth.py:authenticate
        """
        await self._enable_both(monkeypatch, make_authenticator)
        PRINCIPAL.set(Principal(subject="outer-caller"))

        await authenticate(credentials=None, x_api_key="good-key")

        assert PRINCIPAL.get() is None

    async def test_a_principal_never_survives_an_unauthenticated_deployment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same holds with no method configured, where nothing is verified.

        Ref: stdapi/auth.py:authenticate
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", CognitoAuthenticator()
        )
        PRINCIPAL.set(Principal(subject="outer-caller"))

        await authenticate(credentials=None, x_api_key=None)

        assert PRINCIPAL.get() is None

    async def test_pool_token_is_accepted_in_the_api_key_header(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """A token sent as ``x-api-key`` authenticates too, for Anthropic-style clients.

        The Anthropic SDK sends its credential in ``x-api-key``, so a token has
        to be accepted there or the Anthropic-compatible routes could not be
        used with a user pool at all.

        Ref: stdapi/auth.py:authenticate
        """
        await self._enable_both(monkeypatch, make_authenticator)

        await authenticate(
            credentials=None, x_api_key=mint(access_claims(), signing_keys[ACCESS_KID])
        )

        principal = PRINCIPAL.get()
        assert principal is not None
        assert principal.client_id == CLIENT_ID

    async def test_an_api_key_shaped_like_a_token_still_works(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> None:
        """An API key that happens to have three dot-separated parts is not broken.

        Enabling the user pool must not change what the API key accepts, so a
        key that merely looks like a token falls back to the comparison.

        Ref: stdapi/auth.py:authenticate
        """
        harness = await make_authenticator()
        handler = AuthenticationHandler()
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("aa.bb.cc"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        assert await handler.initialize() is True
        monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", harness.authenticator
        )

        await authenticate(credentials=None, x_api_key="aa.bb.cc")

        assert PRINCIPAL.get() is None

    async def test_invalid_token_falls_back_to_the_api_key_and_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        foreign_key: RSAPrivateKey,
    ) -> None:
        """A bearer that is neither a valid token nor the API key is refused once.

        Ref: stdapi/auth.py:authenticate
        """
        await self._enable_both(monkeypatch, make_authenticator)
        token = mint(access_claims(), foreign_key)

        with pytest.raises(ApiError) as excinfo:
            await authenticate(
                credentials=HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=token
                ),
                x_api_key=None,
            )
        assert excinfo.value.status == 401
        assert PRINCIPAL.get() is None

    async def test_invalid_token_is_rejected_when_no_api_key_is_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        foreign_key: RSAPrivateKey,
    ) -> None:
        """With the pool as the only method, a failed verification cannot fall through.

        The API-key handler accepts everything while no key is configured, so a
        fall-back into it would authenticate every request.

        Ref: stdapi/auth.py:authenticate
        """
        harness = await make_authenticator()
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", harness.authenticator
        )

        with pytest.raises(ApiError) as excinfo:
            await authenticate(
                credentials=HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=mint(access_claims(), foreign_key)
                ),
                x_api_key=None,
            )
        assert excinfo.value.status == 401

    async def test_api_key_header_is_rejected_when_only_the_pool_is_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> None:
        """``x-api-key`` cannot authenticate a deployment that configured no API key.

        Ref: stdapi/auth.py:authenticate
        """
        harness = await make_authenticator()
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", harness.authenticator
        )

        with pytest.raises(ApiError) as excinfo:
            await authenticate(credentials=None, x_api_key="anything")
        assert excinfo.value.status == 401

    async def test_no_credentials_is_rejected_when_only_the_pool_is_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> None:
        """An anonymous request is refused as soon as any method is configured.

        Ref: stdapi/auth.py:authenticate
        """
        harness = await make_authenticator()
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", harness.authenticator
        )

        with pytest.raises(ApiError) as excinfo:
            await authenticate(credentials=None, x_api_key=None)
        assert excinfo.value.status == 401

    async def test_token_is_scrubbed_from_the_credentials_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
        signing_keys: dict[str, RSAPrivateKey],
    ) -> None:
        """The bearer value is blanked once consumed, as it is for an API key.

        Ref: stdapi/auth.py:authenticate
        """
        await self._enable_both(monkeypatch, make_authenticator)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=mint(access_claims(), signing_keys[ACCESS_KID])
        )

        await authenticate(credentials=credentials, x_api_key=None)

        assert credentials.credentials == ""


class TestUnauthorizedResponse:
    """What a client actually receives when a token is refused.

    Ref: stdapi/main.py:set_www_authenticate_header
         stdapi/utils.py:hide_security_details
    """

    @pytest.fixture
    async def client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_authenticator: Callable[..., Coroutine[Any, Any, Harness]],
    ) -> TestClient:
        """Lifespan-free client whose only authentication method is the user pool."""
        from stdapi.main import app  # noqa: PLC0415

        harness = await make_authenticator()
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        monkeypatch.setattr(
            stdapi.auth, "_cognito_authenticator", harness.authenticator
        )
        return TestClient(app)

    @pytest.mark.parametrize(
        ("path", "headers"),
        [
            ("/v1/models", {}),
            ("/anthropic/v1/models", {"anthropic-version": "2023-06-01"}),
        ],
    )
    def test_rejected_token_answers_an_opaque_challenge(
        self,
        client: TestClient,
        foreign_key: RSAPrivateKey,
        path: str,
        headers: dict[str, str],
    ) -> None:
        """Both envelopes answer 401, a bearer challenge, and no reason at all.

        The challenge may carry discovery parameters (see
        ``TestWwwAuthenticateChallenge``), but never an ``error`` code: that is
        what would tell a prober which half of the credential was wrong.

        Ref: https://www.rfc-editor.org/rfc/rfc9110.html#name-401-unauthorized
             stdapi/main.py:set_www_authenticate_header
        """
        response = client.get(
            path,
            headers={
                **headers,
                "Authorization": f"Bearer {mint(access_claims(), foreign_key)}",
            },
        )

        assert response.status_code == 401
        challenge = response.headers["www-authenticate"]
        assert challenge == "Bearer" or challenge.startswith("Bearer ")
        assert "error=" not in challenge
        assert "Unauthorized" in response.text
        assert "signature" not in response.text.lower()

    def test_valid_token_reaches_the_route(
        self,
        client: TestClient,
        signing_keys: dict[str, RSAPrivateKey],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An authenticated route answers normally with a valid token, and sends no challenge.

        Ref: https://www.rfc-editor.org/rfc/rfc9110.html
             stdapi/main.py:set_www_authenticate_header
        """

        async def _no_models() -> bool:
            return False

        monkeypatch.setattr(core_models, "initialize_bedrock_models", _no_models)
        response = client.get(
            "/search_models",
            headers={
                "Authorization": f"Bearer {mint(access_claims(), signing_keys[ACCESS_KID])}"
            },
        )

        assert response.status_code == 200
        assert "www-authenticate" not in response.headers
