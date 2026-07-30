"""GET /health, the unauthenticated liveness probe.

Ref: stdapi/routes/core_root.py:health_check
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def client(test_client: TestClient | None) -> TestClient:
    """Return the session test client, skipping if not running locally."""
    if test_client is None:
        pytest.skip("Requires local test server")
    return test_client


class TestHealth:
    """GET /health liveness contract.

    Ref: stdapi/routes/core_root.py:health_check
         stdapi/routes/core_root.py:HealthResponse
    """

    def test_response_body(self, client: TestClient) -> None:
        """GET /health returns HTTP 200 with exactly ``{"status": "ok"}``.

        The route returns the ``HealthResponse`` dataclass whose only field
        defaults to ``"ok"``; FastAPI serialises it with no extra envelope.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_json_content_type(self, client: TestClient) -> None:
        """GET /health is served as ``application/json``."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_no_auth_required(self, client: TestClient) -> None:
        """GET /health succeeds with no credentials and with invalid ones alike.

        The route declares no ``Depends(authenticate)``, so an API key is
        neither required nor validated: a deliberately wrong bearer token must
        not turn into the 401 that authenticated routes return.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = client.get("/health")
        assert anonymous.status_code == 200
        assert anonymous.json() == {"status": "ok"}

        bad_key = client.get("/health", headers={"Authorization": "Bearer wrong-key"})
        assert bad_key.status_code == 200
        assert bad_key.json() == {"status": "ok"}
