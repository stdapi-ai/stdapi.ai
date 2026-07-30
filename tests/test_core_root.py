"""Unauthenticated discovery endpoints: /, /.well-known/api-catalog, the MCP server card and /robots.txt.

Every payload in this module is built once at import time from ``SETTINGS``, so
each test derives its expectation from the settings rather than from the
module-level constant it is meant to police.

Ref: stdapi/routes/core_root.py
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from starlette.testclient import TestClient

#: Documentation target the welcome message points at for the current settings.
_EXPECTED_DOC_TARGET = (
    "/docs"
    if SETTINGS.enable_docs
    else ("/redoc" if SETTINGS.enable_redoc else "https://stdapi.ai/api_reference/")
)

#: Whether at least one MCP transport is enabled, which gates the server card.
_MCP_ENABLED = SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse


@pytest.fixture(scope="module")
def client(test_client: TestClient | None) -> TestClient:
    """Return the session test client, skipping if not running locally."""
    if test_client is None:
        pytest.skip("Requires local test server")
    return test_client


class TestRoot:
    """GET / welcome payload and its RFC 8288 ``Link`` discovery header.

    Ref: stdapi/routes/core_root.py:root
         stdapi/routes/core_root.py:_WELCOME
    """

    def test_returns_200(self, client: TestClient) -> None:
        """GET / returns HTTP 200."""
        assert client.get("/").status_code == 200

    def test_response_has_message(self, client: TestClient) -> None:
        """GET / returns a single ``message`` pointing at the enabled documentation target.

        The pointer is resolved once at import time: ``/docs`` when Swagger UI
        is enabled, otherwise ``/redoc``, otherwise the public documentation URL.
        """
        body = client.get("/").json()
        assert set(body) == {"message"}
        message = body["message"]
        assert isinstance(message, str)
        assert message == (
            "Welcome to the stdapi.ai API! Documentation is available at "
            f"{_EXPECTED_DOC_TARGET}"
        )

    def test_json_content_type(self, client: TestClient) -> None:
        """GET / is served as ``application/json``."""
        response = client.get("/")
        assert "application/json" in response.headers["content-type"]
        assert "message" in response.json()

    def test_no_auth_required(self, client: TestClient) -> None:
        """GET / succeeds with no credentials and with invalid ones alike.

        The metadata router declares no ``Depends(authenticate)``, so a wrong
        bearer token must not produce the 401 authenticated routes return.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = client.get("/")
        assert anonymous.status_code == 200

        bad_key = client.get("/", headers={"Authorization": "Bearer wrong-key"})
        assert bad_key.status_code == 200
        assert bad_key.json() == anonymous.json()

    def test_link_header_absent_when_features_disabled(
        self, client: TestClient
    ) -> None:
        """GET / advertises exactly the enabled discovery resources in its ``Link`` header.

        The header is omitted entirely when none of OpenAPI JSON, Swagger UI /
        ReDoc and MCP are enabled — the default test configuration.

        Ref: stdapi/routes/core_root.py:_LINK_HEADER
        """
        expected_parts = [
            part
            for part in (
                '</openapi.json>; rel="service-desc"'
                if SETTINGS.enable_openapi_json
                else None,
                '</docs>; rel="service-doc"'
                if SETTINGS.enable_docs
                else ('</redoc>; rel="service-doc"' if SETTINGS.enable_redoc else None),
                '</.well-known/mcp/server-card.json>; rel="mcp-server-card"'
                if _MCP_ENABLED
                else None,
            )
            if part is not None
        ]

        response = client.get("/")
        if not expected_parts:
            assert "link" not in response.headers
        else:
            assert response.headers["link"] == ", ".join(expected_parts)


class TestApiCatalog:
    """GET /.well-known/api-catalog serves an RFC 9727 / RFC 9264 linkset.

    Ref: stdapi/routes/core_root.py:api_catalog
         stdapi/routes/core_root.py:_API_CATALOG
    """

    def test_returns_200(self, client: TestClient) -> None:
        """GET /.well-known/api-catalog returns HTTP 200."""
        assert client.get("/.well-known/api-catalog").status_code == 200

    def test_content_type_linkset(self, client: TestClient) -> None:
        """GET /.well-known/api-catalog is served as ``application/linkset+json``."""
        ct = client.get("/.well-known/api-catalog").headers["content-type"]
        assert "application/linkset+json" in ct

    def test_body_has_linkset_key(self, client: TestClient) -> None:
        """The body is a ``linkset`` holding exactly one link context object."""
        body = client.get("/.well-known/api-catalog").json()
        assert set(body) == {"linkset"}
        assert isinstance(body["linkset"], list)
        assert len(body["linkset"]) == 1

    def test_linkset_entry_has_anchor(self, client: TestClient) -> None:
        """The single linkset entry is anchored on the catalog's own path."""
        body = client.get("/.well-known/api-catalog").json()
        entry = body["linkset"][0]
        assert entry["anchor"] == "/.well-known/api-catalog"

    def test_linkset_optional_sections_match_settings(self, client: TestClient) -> None:
        """service-desc, service-doc and mcp-server-card appear only for enabled features.

        Each relation is derived from ``SETTINGS`` here rather than read back
        from ``_API_CATALOG``, so a wrong href or a relation advertised for a
        disabled feature fails the test.
        """
        (entry,) = client.get("/.well-known/api-catalog").json()["linkset"]

        if SETTINGS.enable_openapi_json:
            assert entry["service-desc"] == [
                {
                    "href": "/openapi.json",
                    "type": "application/vnd.oai.openapi+json;version=3.0",
                }
            ]
        else:
            assert "service-desc" not in entry

        if SETTINGS.enable_docs:
            assert entry["service-doc"] == [{"href": "/docs", "title": "Swagger UI"}]
        elif SETTINGS.enable_redoc:
            assert entry["service-doc"] == [{"href": "/redoc", "title": "ReDoc"}]
        else:
            assert "service-doc" not in entry

        if _MCP_ENABLED:
            assert entry["mcp-server-card"] == [
                {"href": "/.well-known/mcp/server-card.json"}
            ]
        else:
            assert "mcp-server-card" not in entry


class TestMcpServerCard:
    """GET /.well-known/mcp/server-card.json is served only when an MCP transport is on.

    Ref: stdapi/routes/core_root.py:mcp_server_card
         stdapi/routes/core_root.py:MCP_SERVER_CARD
    """

    def test_reflects_mcp_status(self, client: TestClient) -> None:
        """The card is 200 when a transport is enabled and a 404 ``error`` payload otherwise."""
        response = client.get("/.well-known/mcp/server-card.json")
        body = response.json()
        if not _MCP_ENABLED:
            assert response.status_code == 404
            assert body == {"error": "MCP is not enabled"}
        else:
            assert response.status_code == 200
            assert body["serverInfo"]["name"]
            assert body["serverInfo"]["version"]
            assert body["capabilities"] == {"tools": {}}
            assert body["transport"]["endpoint"] in {"/mcp", "/sse"}

    def test_no_auth_required(self, client: TestClient) -> None:
        """Credentials never change the card's outcome, and it never answers 401.

        The route is reachable anonymously, so the "not enabled" 404 must not be
        masked by an authentication failure.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = client.get("/.well-known/mcp/server-card.json")
        bad_key = client.get(
            "/.well-known/mcp/server-card.json",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert anonymous.status_code != 401
        assert bad_key.status_code == anonymous.status_code
        assert bad_key.json() == anonymous.json()

    def test_mcp_disabled_returns_error_body(self, client: TestClient) -> None:
        """With every MCP transport off the route answers 404 ``MCP is not enabled``."""
        if _MCP_ENABLED:
            pytest.skip("MCP is enabled in this environment")
        response = client.get("/.well-known/mcp/server-card.json")
        assert response.status_code == 404
        assert response.json() == {"error": "MCP is not enabled"}

    def test_mcp_enabled_body_structure(self, client: TestClient) -> None:
        """The card pins the SEP-1649 schema, version, protocol and the active transport.

        ``streamable-http`` on ``/mcp`` wins when it is enabled; the SSE
        transport on ``/sse`` is only advertised as the fallback.
        """
        if not _MCP_ENABLED:
            pytest.skip("MCP is disabled in this environment")
        from stdapi.metering import EDITION_TITLE  # noqa: PLC0415
        from stdapi.server import SERVER_VERSION  # noqa: PLC0415

        body = client.get("/.well-known/mcp/server-card.json").json()
        assert body["$schema"] == (
            "https://static.modelcontextprotocol.io/schemas/mcp-server-card/v1.json"
        )
        assert body["version"] == "1.0"
        assert body["protocolVersion"] == "2025-03-26"
        assert body["serverInfo"] == {"name": EDITION_TITLE, "version": SERVER_VERSION}
        assert body["transport"] == (
            {"type": "streamable-http", "endpoint": "/mcp"}
            if SETTINGS.enable_mcp_streamable_http
            else {"type": "sse", "endpoint": "/sse"}
        )


class TestRobotsTxt:
    """GET /robots.txt denies crawling by default and opens only the enabled doc paths.

    Ref: stdapi/routes/core_root.py:robots_txt
         stdapi/routes/core_root.py:_ROBOTS_TXT
    """

    def test_returns_200(self, client: TestClient) -> None:
        """GET /robots.txt returns HTTP 200."""
        assert client.get("/robots.txt").status_code == 200

    def test_plain_text_content_type(self, client: TestClient) -> None:
        """GET /robots.txt is served as ``text/plain``."""
        assert "text/plain" in client.get("/robots.txt").headers["content-type"]

    def test_matches_settings(self, client: TestClient) -> None:
        """The record lists exactly the doc paths the current settings enable.

        Built from ``SETTINGS`` rather than compared against ``_ROBOTS_TXT`` so a
        path allowed for a disabled feature — or a missing allow for an enabled
        one — is caught. Ordering is asserted too: the wildcard ``User-agent: *``
        group opens the record, the AI ``Content-Signal`` opts the docs into
        training, search and AI input, and the blanket ``Disallow: /`` comes last
        so the ``Allow:`` lines win under longest-match precedence.
        """
        expected = [
            line
            for line in (
                "User-agent: *",
                "Content-Signal: ai-train=yes, search=yes, ai-input=yes",
                "Allow: /docs" if SETTINGS.enable_docs else None,
                "Allow: /redoc" if SETTINGS.enable_redoc else None,
                "Allow: /openapi.json" if SETTINGS.enable_openapi_json else None,
                "Allow: /.well-known/",
                "Disallow: /",
            )
            if line is not None
        ]
        assert client.get("/robots.txt").text.splitlines() == expected
