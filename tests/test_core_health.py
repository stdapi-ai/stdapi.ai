"""Tests for the /health endpoint."""

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
    """Tests for GET /health."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET /health returns HTTP 200."""
        assert client.get("/health").status_code == 200

    def test_response_body(self, client: TestClient) -> None:
        """GET /health returns {"status": "ok"}."""
        assert client.get("/health").json() == {"status": "ok"}

    def test_json_content_type(self, client: TestClient) -> None:
        """GET /health returns JSON content-type."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_no_auth_required(self, client: TestClient) -> None:
        """GET /health succeeds without an Authorization header."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_repeated_calls_consistent(self, client: TestClient) -> None:
        """GET /health returns the same body across multiple calls."""
        assert client.get("/health").json() == client.get("/health").json()
