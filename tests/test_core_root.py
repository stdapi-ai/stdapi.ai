"""Tests for /, /.well-known/api-catalog, /.well-known/mcp/server-card.json, /robots.txt."""

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


class TestRoot:
    """Tests for GET /."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET / returns HTTP 200."""
        assert client.get("/").status_code == 200

    def test_response_has_message(self, client: TestClient) -> None:
        """GET / response body contains a 'message' key."""
        body = client.get("/").json()
        assert "message" in body
        assert isinstance(body["message"], str)
        assert body["message"]

    def test_json_content_type(self, client: TestClient) -> None:
        """GET / returns JSON content-type."""
        assert "application/json" in client.get("/").headers["content-type"]

    def test_no_auth_required(self, client: TestClient) -> None:
        """GET / does not require an Authorization header."""
        assert client.get("/").status_code == 200

    def test_link_header_absent_when_features_disabled(
        self, client: TestClient
    ) -> None:
        """GET / has no Link header when docs, redoc, openapi and MCP are all disabled.

        The default test configuration does not enable any of those features,
        so _LINK_HEADER is None and the header must be absent.
        """
        from stdapi.routes.core_root import _LINK_HEADER  # noqa: PLC0415

        response = client.get("/")
        if _LINK_HEADER is None:
            assert "link" not in response.headers
        else:
            assert "link" in response.headers
            assert response.headers["link"] == _LINK_HEADER


class TestApiCatalog:
    """Tests for GET /.well-known/api-catalog."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET /.well-known/api-catalog returns HTTP 200."""
        assert client.get("/.well-known/api-catalog").status_code == 200

    def test_content_type_linkset(self, client: TestClient) -> None:
        """GET /.well-known/api-catalog uses application/linkset+json content-type."""
        ct = client.get("/.well-known/api-catalog").headers["content-type"]
        assert "application/linkset+json" in ct

    def test_body_has_linkset_key(self, client: TestClient) -> None:
        """GET /.well-known/api-catalog body has a top-level 'linkset' list."""
        body = client.get("/.well-known/api-catalog").json()
        assert "linkset" in body
        assert isinstance(body["linkset"], list)
        assert len(body["linkset"]) > 0

    def test_linkset_entry_has_anchor(self, client: TestClient) -> None:
        """First linkset entry contains the expected anchor URL."""
        body = client.get("/.well-known/api-catalog").json()
        entry = body["linkset"][0]
        assert "anchor" in entry
        assert entry["anchor"] == "/.well-known/api-catalog"

    def test_linkset_optional_sections_match_settings(self, client: TestClient) -> None:
        """Linkset entry contains service-desc/service-doc/mcp-server-card only when enabled.

        Reads the live _API_CATALOG constant so the assertion mirrors the actual
        server configuration rather than hard-coding assumptions about feature flags.
        """
        from stdapi.routes.core_root import _API_CATALOG  # noqa: PLC0415

        body = client.get("/.well-known/api-catalog").json()
        assert body == _API_CATALOG


class TestMcpServerCard:
    """Tests for GET /.well-known/mcp/server-card.json."""

    def test_reflects_mcp_status(self, client: TestClient) -> None:
        """GET /.well-known/mcp/server-card.json returns 200 when MCP enabled, 404 otherwise."""
        from stdapi.routes.core_root import MCP_SERVER_CARD  # noqa: PLC0415

        response = client.get("/.well-known/mcp/server-card.json")
        if MCP_SERVER_CARD is None:
            assert response.status_code == 404
            assert "error" in response.json()
        else:
            assert response.status_code == 200
            body = response.json()
            assert "serverInfo" in body
            assert "transport" in body
            assert "capabilities" in body

    def test_no_auth_required(self, client: TestClient) -> None:
        """GET /.well-known/mcp/server-card.json does not require authorization."""
        response = client.get("/.well-known/mcp/server-card.json")
        assert response.status_code in {200, 404}

    def test_mcp_disabled_returns_error_body(self, client: TestClient) -> None:
        """When MCP is disabled the response body contains an 'error' key."""
        from stdapi.routes.core_root import MCP_SERVER_CARD  # noqa: PLC0415

        if MCP_SERVER_CARD is not None:
            pytest.skip("MCP is enabled in this environment")
        body = client.get("/.well-known/mcp/server-card.json").json()
        assert "error" in body

    def test_mcp_enabled_body_structure(self, client: TestClient) -> None:
        """When MCP is enabled the server card has the expected schema fields."""
        from stdapi.routes.core_root import MCP_SERVER_CARD  # noqa: PLC0415

        if MCP_SERVER_CARD is None:
            pytest.skip("MCP is disabled in this environment")
        body = client.get("/.well-known/mcp/server-card.json").json()
        assert body["version"] == "1.0"
        assert "protocolVersion" in body
        assert body["transport"]["type"] in {"streamable-http", "sse"}


class TestRobotsTxt:
    """Tests for GET /robots.txt."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET /robots.txt returns HTTP 200."""
        assert client.get("/robots.txt").status_code == 200

    def test_plain_text_content_type(self, client: TestClient) -> None:
        """GET /robots.txt returns plain text content-type."""
        assert "text/plain" in client.get("/robots.txt").headers["content-type"]

    def test_has_user_agent_directive(self, client: TestClient) -> None:
        """GET /robots.txt contains a User-agent directive."""
        assert "User-agent: *" in client.get("/robots.txt").text

    def test_has_disallow_all(self, client: TestClient) -> None:
        """GET /robots.txt contains 'Disallow: /' to restrict crawlers."""
        assert "Disallow: /" in client.get("/robots.txt").text

    def test_allows_well_known(self, client: TestClient) -> None:
        """GET /robots.txt always allows /.well-known/."""
        assert "Allow: /.well-known/" in client.get("/robots.txt").text

    def test_has_ai_content_signal(self, client: TestClient) -> None:
        """GET /robots.txt contains the AI content signal header."""
        assert "Content-Signal:" in client.get("/robots.txt").text

    def test_matches_settings(self, client: TestClient) -> None:
        """GET /robots.txt exactly matches the _ROBOTS_TXT constant."""
        from stdapi.routes.core_root import _ROBOTS_TXT  # noqa: PLC0415

        assert client.get("/robots.txt").text == _ROBOTS_TXT
