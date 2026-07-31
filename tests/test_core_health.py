"""GET /health, the unauthenticated liveness probe.

Ref: stdapi/routes/core_root.py:health_check
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from starlette.testclient import TestClient

pytestmark = pytest.mark.local


class TestHealth:
    """GET /health liveness contract.

    Ref: stdapi/routes/core_root.py:health_check
         stdapi/routes/core_root.py:HealthResponse
    """

    def test_response_body(self, test_client: TestClient) -> None:
        """GET /health returns HTTP 200 with exactly ``{"status": "ok"}``.

        The route returns the ``HealthResponse`` dataclass whose only field
        defaults to ``"ok"``; FastAPI serialises it with no extra envelope.
        """
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_json_content_type(self, test_client: TestClient) -> None:
        """GET /health is served as ``application/json``."""
        response = test_client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_no_auth_required(self, test_client: TestClient) -> None:
        """GET /health succeeds with no credentials and with invalid ones alike.

        The route declares no ``Depends(authenticate)``, so an API key is
        neither required nor validated: a deliberately wrong bearer token must
        not turn into the 401 that authenticated routes return.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = test_client.get("/health")
        assert anonymous.status_code == 200
        assert anonymous.json() == {"status": "ok"}

        bad_key = test_client.get(
            "/health", headers={"Authorization": "Bearer wrong-key"}
        )
        assert bad_key.status_code == 200
        assert bad_key.json() == {"status": "ok"}
