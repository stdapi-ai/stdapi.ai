"""Amazon Cognito user pool authentication against a real user pool.

Every other pool test signs its tokens with a key it generated and answers the
key-set fetch itself, so nothing in them depends on a pool existing. This module
is the counterpart: it obtains a token from a real pool with the
``client_credentials`` grant and lets the gateway verify it, which is the only
thing that shows the published key set parses, that a real access token carries
the claims the verification reads, and that the pool's endpoints are reachable
at all.

The pool is created by the ``terraform-sandbox`` stack; a checkout without one
skips.

Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/token-endpoint.html
     stdapi/auth_cognito.py:CognitoAuthenticator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from starlette.testclient import TestClient

import stdapi.auth
from stdapi.auth import AuthenticationHandler
from stdapi.auth_cognito import CognitoAuthenticator
from stdapi.config import SETTINGS
from stdapi.routes import core_models

if TYPE_CHECKING:
    from tests.conftest import CognitoSandboxPool

pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]

#: Seconds a token request gets before it counts as the pool being unreachable.
_TOKEN_TIMEOUT = 30.0


def _mint(pool: CognitoSandboxPool, client_id: str, client_secret: str) -> str:
    """Obtain an access token from the pool the way a machine client does.

    Args:
        pool: The pool to ask, and the scope to ask for.
        client_id: Application requesting the token.
        client_secret: That application's secret.

    Returns:
        The access token the pool issued.
    """
    response = httpx.post(
        pool.token_url,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials", "scope": pool.scope},
        timeout=_TOKEN_TIMEOUT,
    )
    assert response.status_code == 200, (
        f"The user pool refused to issue a token to '{client_id}': {response.text}"
    )
    return str(response.json()["access_token"])


@pytest.fixture(scope="module")
def access_token(cognito_sandbox_pool: CognitoSandboxPool) -> str:
    """Access token of the application the deployment accepts, minted once."""
    return _mint(
        cognito_sandbox_pool,
        cognito_sandbox_pool.client_id,
        cognito_sandbox_pool.client_secret,
    )


@pytest.fixture(scope="module")
def foreign_access_token(cognito_sandbox_pool: CognitoSandboxPool) -> str:
    """Access token of the same pool's other application, minted once."""
    return _mint(
        cognito_sandbox_pool,
        cognito_sandbox_pool.foreign_client_id,
        cognito_sandbox_pool.foreign_client_secret,
    )


@pytest.fixture
async def authenticator(
    monkeypatch: pytest.MonkeyPatch, cognito_sandbox_pool: CognitoSandboxPool
) -> CognitoAuthenticator:
    """An authenticator that has loaded the real pool's published signing keys."""
    monkeypatch.setattr(
        SETTINGS, "aws_cognito_user_pool_id", cognito_sandbox_pool.user_pool_id
    )
    monkeypatch.setattr(
        SETTINGS, "aws_cognito_client_ids", [cognito_sandbox_pool.client_id]
    )
    monkeypatch.setattr(
        SETTINGS, "aws_cognito_required_scopes", [cognito_sandbox_pool.scope]
    )
    monkeypatch.setattr(SETTINGS, "aws_cognito_accept_id_token", False)
    monkeypatch.setattr(SETTINGS, "aws_cognito_issuer_type", "original")
    authenticator = CognitoAuthenticator()
    assert await authenticator.initialize() is True
    return authenticator


@pytest.fixture
def gateway(
    monkeypatch: pytest.MonkeyPatch, authenticator: CognitoAuthenticator
) -> TestClient:
    """Lifespan-free gateway whose only accepted credential is a pool token."""
    from stdapi.main import app  # noqa: PLC0415

    async def _no_models() -> bool:
        return False

    monkeypatch.setattr(core_models, "initialize_bedrock_models", _no_models)
    monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
    monkeypatch.setattr(stdapi.auth, "_cognito_authenticator", authenticator)
    return TestClient(app)


class TestLiveUserPool:
    """What a token a real pool issued proves that a minted one cannot.

    Ref: stdapi/auth_cognito.py:CognitoAuthenticator.verify
    """

    async def test_a_real_access_token_carries_the_claims_the_gateway_reads(
        self,
        authenticator: CognitoAuthenticator,
        access_token: str,
        cognito_sandbox_pool: CognitoSandboxPool,
    ) -> None:
        """The pool's own token verifies, and identifies the application and scope.

        A machine-to-machine token names its application in ``client_id`` and
        carries no user, so a verification reading ``aud`` or requiring a
        username would fail here while passing on every locally minted token.

        Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
             stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        principal = await authenticator.verify(access_token)

        assert principal.client_id == cognito_sandbox_pool.client_id
        assert cognito_sandbox_pool.scope in principal.scopes
        assert principal.subject
        assert principal.username is None

    def test_a_real_access_token_authenticates_a_gateway_request(
        self, gateway: TestClient, access_token: str
    ) -> None:
        """A request bearing that token reaches the route, unchallenged.

        Ref: stdapi/auth.py:verify_credential
        """
        response = gateway.get(
            "/search_models", headers={"Authorization": f"Bearer {access_token}"}
        )

        assert response.status_code == 200
        assert "www-authenticate" not in response.headers

    def test_a_token_from_another_application_is_rejected(
        self, gateway: TestClient, foreign_access_token: str
    ) -> None:
        """The same pool's other application is refused, since it is not allowlisted.

        It differs from the accepted token only in its ``client_id``: same
        issuer, same signing key, same scope. Without the application check, a
        pool that allows anyone to register an application would let anyone call
        the API.

        Ref: stdapi/auth_cognito.py:CognitoAuthenticator._principal
        """
        response = gateway.get(
            "/search_models",
            headers={"Authorization": f"Bearer {foreign_access_token}"},
        )

        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert "Unauthorized" in response.text
