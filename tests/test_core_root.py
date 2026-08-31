"""Unauthenticated discovery endpoints: /, the well-known documents, the MCP server card and /robots.txt.

Every payload in this module is built once at import time from ``SETTINGS``, so
each test derives its expectation from the settings rather than from the
module-level constant it is meant to police.

Ref: stdapi/routes/core_root.py
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from pathlib import Path

    import httpx
    from starlette.testclient import TestClient

#: Campaign parameters every link a human clicks off the gateway carries.
_DOCS_UTM = "utm_source=api-docs&utm_medium=product&utm_campaign=owned-surfaces"

#: Documentation target the welcome message points at for the current settings.
_EXPECTED_DOC_TARGET = (
    "/docs"
    if SETTINGS.enable_docs
    else (
        "/redoc"
        if SETTINGS.enable_redoc
        else f"https://stdapi.ai/api_reference/?{_DOCS_UTM}"
    )
)

#: Whether at least one MCP transport is enabled, which gates the server card.
_MCP_ENABLED = SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse

#: Path RFC 9728 reserves for the protected resource metadata of a root-mounted API.
_OAUTH_METADATA_PATH = "/.well-known/oauth-protected-resource"

#: Whether the protected resource metadata document is published at all.
_OAUTH_ENABLED = bool(SETTINGS.oauth_resource_identifier)

#: Link relation naming the metadata document; RFC 9264 requires an extension one to be a URI.
_OAUTH_RELATION = "https://www.rfc-editor.org/rfc/rfc9728"

pytestmark = pytest.mark.local


class TestRoot:
    """GET / welcome payload and its RFC 8288 ``Link`` discovery header.

    Ref: stdapi/routes/core_root.py:root
         stdapi/routes/core_root.py:_WELCOME
    """

    def test_returns_200(self, test_client: TestClient) -> None:
        """GET / returns HTTP 200."""
        assert test_client.get("/").status_code == 200

    def test_response_has_message(self, test_client: TestClient) -> None:
        """GET / returns a single ``message`` pointing at the enabled documentation target.

        The pointer is resolved once at import time: ``/docs`` when Swagger UI
        is enabled, otherwise ``/redoc``, otherwise the public documentation URL.
        """
        body = test_client.get("/").json()
        assert set(body) == {"message"}
        message = body["message"]
        assert isinstance(message, str)
        assert message == (
            "Welcome to the stdapi.ai API! Documentation is available at "
            f"{_EXPECTED_DOC_TARGET}"
        )

    def test_the_public_documentation_link_is_campaign_tagged(
        self, test_client: TestClient
    ) -> None:
        """The welcome message's public URL carries the campaign parameters.

        The fallback is only reached with both built-in doc pages disabled, and
        it is the one link in this payload a human clicks: untagged, its traffic
        cannot be told apart from any other arrival at the site.

        Ref: stdapi/routes/core_root.py:_API_REFERENCE_LINK
             stdapi/metering.py:DOCS_UTM
        """
        if SETTINGS.enable_docs or SETTINGS.enable_redoc:
            pytest.skip("A built-in documentation page is enabled")

        assert _DOCS_UTM in test_client.get("/").json()["message"]

    def test_json_content_type(self, test_client: TestClient) -> None:
        """GET / is served as ``application/json``."""
        response = test_client.get("/")
        assert "application/json" in response.headers["content-type"]
        assert "message" in response.json()

    def test_no_auth_required(self, test_client: TestClient) -> None:
        """GET / succeeds with no credentials and with invalid ones alike.

        The metadata router declares no ``Depends(authenticate)``, so a wrong
        bearer token must not produce the 401 authenticated routes return.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = test_client.get("/")
        assert anonymous.status_code == 200

        bad_key = test_client.get("/", headers={"Authorization": "Bearer wrong-key"})
        assert bad_key.status_code == 200
        assert bad_key.json() == anonymous.json()

    def test_link_header_absent_when_features_disabled(
        self, app_client: TestClient
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
                f'<{_OAUTH_METADATA_PATH}>; rel="{_OAUTH_RELATION}"'
                if _OAUTH_ENABLED
                else None,
            )
            if part is not None
        ]

        response = app_client.get("/")
        if not expected_parts:
            assert "link" not in response.headers
        else:
            assert response.headers["link"] == ", ".join(expected_parts)


class TestOpenApiInfoLinks:
    """The two stdapi.ai URLs the OpenAPI ``info`` block carries.

    One is read by a human -- Swagger UI renders ``contact.url`` as the link in
    its header -- and one is read by licence tooling. Only the first is tagged.

    Ref: stdapi/main.py
         stdapi/metering.py:DOCS_UTM
         https://spec.openapis.org/oas/v3.1.0#info-object
    """

    @staticmethod
    def _info() -> dict[str, object]:
        """Return the ``info`` block of the application's own OpenAPI document.

        Read from the application rather than over ``/openapi.json``: the
        document is only served when ``enable_openapi_json`` is set, and it is
        off in the default test configuration.

        Returns:
            The ``info`` object.
        """
        from stdapi.main import app  # noqa: PLC0415

        info: dict[str, object] = app.openapi()["info"]
        return info

    def test_the_contact_url_is_campaign_tagged(self) -> None:
        """``contact.url`` carries the campaign parameters."""
        contact = self._info()["contact"]
        assert isinstance(contact, dict)

        assert contact["url"] == f"https://stdapi.ai/?{_DOCS_UTM}"

    def test_the_licence_url_is_not_tagged(self) -> None:
        """``license.url`` stays clean on both editions: it identifies a licence, not a visit.

        Licence tooling matches this field against a known URL, so a query
        string on it makes the licence unrecognisable. The community build
        never carries a ``url`` at all, so that branch alone would pass
        vacuously; ``PRODUCT_CODE`` is fixed at import time, so the
        commercial branch is built directly through ``licence_info`` instead
        of relying on the process this suite runs in.
        """
        from stdapi.metering import licence_info  # noqa: PLC0415

        community = self._info().get("license", {})
        assert isinstance(community, dict)
        assert "utm_" not in community.get("url", "")

        commercial = licence_info("some-marketplace-product-code")
        assert "utm_" not in commercial.get("url", "")

    def test_the_docs_site_export_carries_no_tracking_query(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The docs site's own copy of the OpenAPI document is not self-tagged.

        ``docs_hooks/fastapi_openapi.py`` exports ``app.openapi()`` into
        ``docs/openapi.yml``, which ``docs/api_reference.md`` renders with
        ReDoc: a tagged ``contact.url`` there would have the docs site link
        to itself with the campaign parameters meant for a visitor arriving
        from elsewhere, overwriting their real traffic source.

        Ref: docs_hooks/fastapi_openapi.py:on_pre_build
        """
        from docs_hooks.fastapi_openapi import on_pre_build  # noqa: PLC0415

        (tmp_path / "docs").mkdir()
        monkeypatch.chdir(tmp_path)

        on_pre_build(None)

        content = (tmp_path / "docs" / "openapi.yml").read_text()
        assert "utm_" not in content


class TestApiCatalog:
    """GET /.well-known/api-catalog serves an RFC 9727 / RFC 9264 linkset.

    Ref: stdapi/routes/core_root.py:api_catalog
         stdapi/routes/core_root.py:_API_CATALOG
    """

    def test_returns_200(self, test_client: TestClient) -> None:
        """GET /.well-known/api-catalog returns HTTP 200."""
        assert test_client.get("/.well-known/api-catalog").status_code == 200

    def test_content_type_linkset(self, test_client: TestClient) -> None:
        """GET /.well-known/api-catalog is served as ``application/linkset+json``."""
        ct = test_client.get("/.well-known/api-catalog").headers["content-type"]
        assert "application/linkset+json" in ct

    def test_body_has_linkset_key(self, test_client: TestClient) -> None:
        """The body is a ``linkset`` holding exactly one link context object."""
        body = test_client.get("/.well-known/api-catalog").json()
        assert set(body) == {"linkset"}
        assert isinstance(body["linkset"], list)
        assert len(body["linkset"]) == 1

    def test_linkset_entry_has_anchor(self, test_client: TestClient) -> None:
        """The single linkset entry is anchored on the catalog's own path."""
        body = test_client.get("/.well-known/api-catalog").json()
        entry = body["linkset"][0]
        assert entry["anchor"] == "/.well-known/api-catalog"

    def test_linkset_optional_sections_match_settings(
        self, app_client: TestClient
    ) -> None:
        """service-desc, service-doc and mcp-server-card appear only for enabled features.

        Each relation is derived from ``SETTINGS`` here rather than read back
        from ``_API_CATALOG``, so a wrong href or a relation advertised for a
        disabled feature fails the test.
        """
        (entry,) = app_client.get("/.well-known/api-catalog").json()["linkset"]

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

        if _OAUTH_ENABLED:
            assert entry[_OAUTH_RELATION] == [{"href": _OAUTH_METADATA_PATH}]
        else:
            assert _OAUTH_RELATION not in entry


class TestMcpServerCard:
    """GET /.well-known/mcp/server-card.json is served only when an MCP transport is on.

    Ref: stdapi/routes/core_root.py:mcp_server_card
         stdapi/routes/core_root.py:MCP_SERVER_CARD
    """

    def test_reflects_mcp_status(self, test_client: TestClient) -> None:
        """The card is 200 when a transport is enabled and a 404 ``error`` payload otherwise."""
        response = test_client.get("/.well-known/mcp/server-card.json")
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

    def test_no_auth_required(self, test_client: TestClient) -> None:
        """Credentials never change the card's outcome, and it never answers 401.

        The route is reachable anonymously, so the "not enabled" 404 must not be
        masked by an authentication failure.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = test_client.get("/.well-known/mcp/server-card.json")
        bad_key = test_client.get(
            "/.well-known/mcp/server-card.json",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert anonymous.status_code != 401
        assert bad_key.status_code == anonymous.status_code
        assert bad_key.json() == anonymous.json()

    def test_mcp_disabled_returns_error_body(self, test_client: TestClient) -> None:
        """With every MCP transport off the route answers 404 ``MCP is not enabled``."""
        if _MCP_ENABLED:
            pytest.skip("MCP is enabled in this environment")
        response = test_client.get("/.well-known/mcp/server-card.json")
        assert response.status_code == 404
        assert response.json() == {"error": "MCP is not enabled"}

    def test_mcp_enabled_body_structure(self, test_client: TestClient) -> None:
        """The card pins the SEP-1649 schema, version, protocol and the active transport.

        ``streamable-http`` on ``/mcp`` wins when it is enabled; the SSE
        transport on ``/sse`` is only advertised as the fallback.
        """
        if not _MCP_ENABLED:
            pytest.skip("MCP is disabled in this environment")
        from stdapi.metering import EDITION_TITLE  # noqa: PLC0415
        from stdapi.server import SERVER_VERSION  # noqa: PLC0415

        body = test_client.get("/.well-known/mcp/server-card.json").json()
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

    def test_returns_200(self, test_client: TestClient) -> None:
        """GET /robots.txt returns HTTP 200."""
        assert test_client.get("/robots.txt").status_code == 200

    def test_plain_text_content_type(self, test_client: TestClient) -> None:
        """GET /robots.txt is served as ``text/plain``."""
        assert "text/plain" in test_client.get("/robots.txt").headers["content-type"]

    def test_matches_settings(self, test_client: TestClient) -> None:
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
        assert test_client.get("/robots.txt").text.splitlines() == expected


class TestErrorsOnLogExemptPaths:
    """HTTP errors on log-exempt paths are answered cleanly, not crashed on.

    Paths in ``LOGGING_PATHS_IGNORE`` run without a request-log context, and
    the error handlers call :func:`~stdapi.monitoring.log_error_details`; the
    logger must therefore tolerate the missing context. A deployment reaches
    this daily: the exempt list names the documentation pages, which most
    deployments leave disabled, and something asks for them anyway.

    Ref: stdapi/monitoring.py:log_error_details
         stdapi/main.py:handle_http_exception
    """

    def test_404_on_an_exempt_path_is_a_clean_error_envelope(
        self, test_client: TestClient
    ) -> None:
        """GET an exempt path with no route returns the JSON 404 envelope, not a 500.

        ``TestClient`` re-raises server exceptions, so a ``LookupError`` in the
        exception handler would fail this test rather than surface as a 500.
        """
        if SETTINGS.enable_docs:
            pytest.skip("Swagger UI is mounted in this configuration")

        response = test_client.get("/docs")
        assert response.status_code == 404
        assert response.json()["error"] == "Not Found"

    def test_method_not_allowed_on_exempt_path_is_clean(
        self, test_client: TestClient
    ) -> None:
        """DELETE /health returns the 405 envelope through the same handler."""
        response = test_client.delete("/health")
        assert response.status_code == 405
        assert response.json()["error"]


class TestOAuthProtectedResource:
    """GET /.well-known/oauth-protected-resource serves RFC 9728 metadata.

    The document is what lets an agent that has never been configured for this
    deployment find out where to obtain a token. It is built once at import
    time, so every expectation below is derived from ``SETTINGS`` instead of
    being read back from the module constant.

    Ref: https://www.rfc-editor.org/rfc/rfc9728.html
         stdapi/routes/core_root.py:oauth_protected_resource
    """

    def test_reflects_configuration(self, app_client: TestClient) -> None:
        """The document is 200 when a resource identifier is set, and 404 otherwise."""
        response = app_client.get(_OAUTH_METADATA_PATH)
        if _OAUTH_ENABLED:
            assert response.status_code == 200
        else:
            assert response.status_code == 404
            assert response.json() == {
                "error": "OAuth 2.0 protected resource metadata is not configured"
            }

    def test_json_content_type(self, app_client: TestClient) -> None:
        """RFC 9728 section 3.2 requires ``application/json`` on the 200 response."""
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        response = app_client.get(_OAUTH_METADATA_PATH)
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_members_match_settings(self, app_client: TestClient) -> None:
        """Every published member comes from the settings that describe the deployment.

        ``resource`` is compared verbatim: RFC 9728 section 3.3 has the client
        compare it against the identifier it inserted the well-known suffix
        into, character by character, so a normalised or slash-suffixed value
        would be rejected.

        The member set is pinned as well: the document is served unauthenticated
        to anyone, so a member added later — a pool identifier, a client
        identifier, a JWKS location — would disclose the deployment's identity
        configuration to the whole internet.
        """
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        from stdapi.metering import EDITION_TITLE  # noqa: PLC0415

        body = app_client.get(_OAUTH_METADATA_PATH).json()

        assert set(body) == {
            "resource",
            "authorization_servers",
            "bearer_methods_supported",
            "resource_name",
            "resource_documentation",
        } | ({"scopes_supported"} if SETTINGS.oauth_scopes_supported else set())
        assert body["resource"] == SETTINGS.oauth_resource_identifier
        assert not body["resource"].endswith("/")
        assert body["authorization_servers"] == SETTINGS.oauth_authorization_servers
        assert body["bearer_methods_supported"] == ["header"]
        assert body["resource_name"] == EDITION_TITLE
        assert body["resource_documentation"] == "https://stdapi.ai/api_reference/"
        if SETTINGS.oauth_scopes_supported:
            assert body["scopes_supported"] == SETTINGS.oauth_scopes_supported
        else:
            assert "scopes_supported" not in body

    def test_resource_ignores_the_request_origin(
        self, app_client: TestClient, enforced_auth_client: TestClient
    ) -> None:
        """A spoofed ``Host`` never changes the resource, nor the URL the challenge names.

        The identifier is the deployment's own; deriving it from the request
        would let anyone who can set the header publish an origin of their
        choosing and collect the tokens an agent then sends there.
        """
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        spoofed = {"Host": "attacker.example"}

        body = app_client.get(_OAUTH_METADATA_PATH, headers=spoofed).json()
        assert body["resource"] == SETTINGS.oauth_resource_identifier

        challenged = enforced_auth_client.get("/v1/models", headers=spoofed)
        assert challenged.status_code == 401
        challenge = challenged.headers["www-authenticate"]
        assert SETTINGS.oauth_resource_identifier is not None
        assert SETTINGS.oauth_resource_identifier in challenge
        assert "attacker.example" not in challenge

    def test_no_zero_valued_member_is_published(self, app_client: TestClient) -> None:
        """RFC 9728 section 3.2: "Parameters with zero values MUST be omitted"."""
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        response = app_client.get(_OAUTH_METADATA_PATH)
        assert response.status_code == 200
        body = response.json()
        assert "resource" in body
        assert all(value for value in body.values()), body

    def test_cacheable(self, app_client: TestClient) -> None:
        """RFC 9728 section 7.10 asks for a cache lifetime on the document."""
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        cache_control = app_client.get(_OAUTH_METADATA_PATH).headers["cache-control"]
        assert "max-age=" in cache_control

    def test_no_auth_required(self, enforced_auth_client: TestClient) -> None:
        """Credentials never change the document, and it never answers 401.

        Bootstrapping starts from a 401, so a client reads this document with no
        usable credential in hand; requiring one would make it unreachable. The
        client here faces an armed API key check — proven by the authenticated
        route refusing it — so the document is read exactly as a credential-less
        agent reads it against a deployment that enforces authentication.

        Ref: stdapi/auth.py:authenticate
        """
        assert enforced_auth_client.get("/v1/models").status_code == 401

        anonymous = enforced_auth_client.get(_OAUTH_METADATA_PATH)
        bad_key = enforced_auth_client.get(
            _OAUTH_METADATA_PATH, headers={"Authorization": "Bearer wrong-key"}
        )
        assert anonymous.status_code != 401
        assert bad_key.status_code == anonymous.status_code
        assert bad_key.json() == anonymous.json()

    def test_accepted_by_the_mcp_client_sdk(self, app_client: TestClient) -> None:
        """The served document validates against the model a real MCP client parses it with.

        The SDK requires at least one ``authorization_servers`` entry, so this
        pins conformance to the client rather than to our reading of the RFC.

        Ref: mcp/shared/auth.py:ProtectedResourceMetadata
        """
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        from mcp.shared.auth import ProtectedResourceMetadata  # noqa: PLC0415

        metadata = ProtectedResourceMetadata.model_validate_json(
            app_client.get(_OAUTH_METADATA_PATH).content
        )
        assert len(metadata.authorization_servers) >= 1

    def test_resource_covers_the_urls_clients_dial(
        self, app_client: TestClient
    ) -> None:
        """The published ``resource`` matches the MCP endpoints an agent connects to.

        Both SDKs compare the origin exactly and require the published resource
        to be a prefix of the dialled URL, so an origin-level identifier has to
        cover ``/mcp`` and ``/sse`` while still rejecting another host. The value
        is read out of the served document, so publishing a path-bearing or
        rewritten identifier fails here.

        Ref: mcp/shared/auth_utils.py:check_resource_allowed
        """
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        from mcp.shared.auth_utils import check_resource_allowed  # noqa: PLC0415

        resource = app_client.get(_OAUTH_METADATA_PATH).json()["resource"]
        for path in ("", "/mcp", "/sse", "/v1/models"):
            assert check_resource_allowed(
                requested_resource=f"{resource}{path}", configured_resource=resource
            )
        assert not check_resource_allowed(
            requested_resource="https://elsewhere.example.com/mcp",
            configured_resource=resource,
        )

    def test_absent_when_no_resource_identifier_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the identifier unset the route answers 404 and the catalog drops the link.

        A hollow document is worse than none: the TypeScript MCP SDK treats a
        missing ``authorization_servers`` as "the gateway is its own
        authorization server" and walks the user to a URL that does not exist,
        while a 404 is the one status both SDKs' fallback chains handle.

        Every payload is built at import time, so the module is re-executed
        under the unset setting -- as a private copy rather than a reload, which
        would replace the live application's own payloads with these.
        """
        from importlib.util import find_spec, module_from_spec  # noqa: PLC0415

        from fastapi import FastAPI  # noqa: PLC0415
        from starlette.testclient import TestClient  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "oauth_resource_identifier", None)
        spec = find_spec("stdapi.routes.core_root")
        assert spec is not None
        assert spec.loader is not None
        unconfigured = module_from_spec(spec)
        spec.loader.exec_module(unconfigured)

        assert unconfigured.WWW_AUTHENTICATE_CHALLENGE == "Bearer"
        app = FastAPI()
        app.include_router(unconfigured.router)
        client = TestClient(app)

        response = client.get(_OAUTH_METADATA_PATH)
        assert response.status_code == 404
        assert response.json() == {
            "error": "OAuth 2.0 protected resource metadata is not configured"
        }
        catalog = client.get("/.well-known/api-catalog").json()
        assert _OAUTH_RELATION not in catalog["linkset"][0]
        assert "oauth-protected-resource" not in (
            client.get("/").headers.get("link") or ""
        )


class TestWwwAuthenticateChallenge:
    """Every 401 carries the ``WWW-Authenticate`` challenge an agent bootstraps from.

    RFC 9110 section 15.5.2 makes the header mandatory on a 401, and RFC 9728
    section 5.1 puts the metadata location in it. The challenge is uniform: it
    never says why a credential was refused, so a missing and a wrong one stay
    indistinguishable.

    Ref: https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.2
         stdapi/main.py:set_www_authenticate_header
    """

    #: A route that authenticates, and one that does not exist on any surface.
    _AUTHENTICATED_PATH = "/v1/models"

    #: Each MCP mount, the method its 401 is met on, and whether it is enabled here.
    _MCP_MOUNTS = (
        ("/mcp", "POST", SETTINGS.enable_mcp_streamable_http),
        ("/sse", "GET", SETTINGS.enable_mcp_sse),
    )

    @staticmethod
    def _expected_challenge() -> str:
        """Rebuild the challenge from the settings the deployment was given."""
        if not _OAUTH_ENABLED:
            return "Bearer"
        metadata_url = f"{SETTINGS.oauth_resource_identifier}{_OAUTH_METADATA_PATH}"
        parameters = [f'resource_metadata="{metadata_url}"']
        if SETTINGS.oauth_scopes_supported:
            parameters.append(f'scope="{" ".join(SETTINGS.oauth_scopes_supported)}"')
        return f"Bearer {', '.join(parameters)}"

    def test_missing_and_wrong_credentials_get_the_same_challenge(
        self, enforced_auth_client: TestClient
    ) -> None:
        """A 401 always carries the challenge, and it is identical either way.

        RFC 6750 section 3 asks a 401 for a credential-less request to carry no
        ``error`` code; keeping it off both responses is also what stops the
        header from telling a prober whether a key exists.
        """
        expected = self._expected_challenge()

        missing = enforced_auth_client.get(self._AUTHENTICATED_PATH)
        wrong = enforced_auth_client.get(
            self._AUTHENTICATED_PATH, headers={"Authorization": "Bearer wrong-key"}
        )

        assert missing.status_code == 401
        assert wrong.status_code == 401
        assert missing.headers["www-authenticate"] == expected
        assert wrong.headers["www-authenticate"] == expected
        assert "error=" not in expected

    @pytest.mark.parametrize(("path", "method", "enabled"), _MCP_MOUNTS)
    def test_carried_by_the_mcp_transport_too(
        self, enforced_auth_client: TestClient, path: str, method: str, enabled: bool
    ) -> None:
        """Each MCP mount answers 401 with the same challenge as the REST routes.

        An MCP client only ever meets the gateway through its mount, so a
        challenge missing here leaves it with nothing to bootstrap from. Both
        mounts are checked against ``LOGGING_PATHS_IGNORE`` whatever the enabled
        transports are: an exempt path skips the setter entirely, which is how a
        mount would lose its challenge without any route changing.

        Ref: stdapi/monitoring.py:LOGGING_PATHS_IGNORE
        """
        from stdapi.monitoring import LOGGING_PATHS_IGNORE  # noqa: PLC0415

        assert path not in LOGGING_PATHS_IGNORE
        if not enabled:
            pytest.skip(f"The MCP transport mounted on {path} is disabled")

        response = enforced_auth_client.request(
            method,
            path,
            json=(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                if method == "POST"
                else None
            ),
            headers={"Accept": "application/json, text/event-stream"},
        )

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == self._expected_challenge()

    def test_absent_on_responses_that_are_not_401(
        self, enforced_auth_client: TestClient
    ) -> None:
        """A 404 carries no challenge, so the header stays a credential signal.

        The path is deliberately one that is not log-exempt: an exempt path
        never reaches the setter, so it could not police the status guard.
        """
        response = enforced_auth_client.get("/v1/does-not-exist")
        assert response.status_code == 404
        assert "www-authenticate" not in response.headers

    def test_readable_by_a_browser_hosted_client(
        self, enforced_auth_client: TestClient
    ) -> None:
        """A cross-origin 401 marks the challenge as readable by the browser.

        ``www-authenticate`` is not a CORS-safelisted response header, so a
        client running in a page — an allowed origin included — reads ``null``
        from it unless the response names it as exposed, and loses both the
        metadata location and the scopes it has to ask a token for.

        Ref: https://www.rfc-editor.org/rfc/rfc9728.html
        """
        if not SETTINGS.cors_allow_origins:
            pytest.skip("No CORS origin is allowed in this environment")

        response = enforced_auth_client.get(
            self._AUTHENTICATED_PATH, headers={"Origin": "https://client.example"}
        )

        assert response.status_code == 401
        exposed = {
            header.strip().lower()
            for header in response.headers["access-control-expose-headers"].split(",")
        }
        assert "www-authenticate" in exposed

    def test_read_by_the_mcp_client_sdk(self, enforced_auth_client: TestClient) -> None:
        """A real MCP client extracts the metadata URL and the scopes from the header.

        The parameter values must be quoted: RFC 9110's ``tchar`` set excludes
        ``:`` and ``/``, and the SDK's deprecated extractor only matches the
        quoted form.

        Ref: mcp/client/auth/utils.py:extract_resource_metadata_from_www_auth
        """
        if not _OAUTH_ENABLED:
            pytest.skip("No OAuth resource identifier configured")
        from mcp.client.auth.utils import (  # noqa: PLC0415
            extract_resource_metadata_from_www_auth,
            extract_scope_from_www_auth,
        )

        # Starlette types TestClient against httpx2; the alias fixes runtime only.
        response: httpx.Response = enforced_auth_client.get(  # type: ignore[assignment]
            self._AUTHENTICATED_PATH
        )

        assert extract_resource_metadata_from_www_auth(response) == (
            f"{SETTINGS.oauth_resource_identifier}{_OAUTH_METADATA_PATH}"
        )
        if SETTINGS.oauth_scopes_supported:
            assert extract_scope_from_www_auth(response) == " ".join(
                SETTINGS.oauth_scopes_supported
            )
