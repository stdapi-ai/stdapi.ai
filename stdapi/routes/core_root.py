"""Root endpoint for the API."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from stdapi.config import SETTINGS
from stdapi.metering import EDITION_TITLE
from stdapi.server import SERVER_VERSION

if TYPE_CHECKING:
    from typing import Any

router = APIRouter(tags=["metadata"], include_in_schema=False)


@dataclass(slots=True, frozen=True)
class HealthResponse:
    """Response for the /health endpoint."""

    status: str = "ok"


@dataclass(slots=True, frozen=True)
class PingResponse:
    """Response for the /ping endpoint."""

    status: str = "Healthy"


#: Welcome message payload for root endpoint
_WELCOME = {
    "message": "Welcome to the stdapi.ai API! Documentation is available at "
    + (
        "/docs"
        if SETTINGS.enable_docs
        else ("/redoc" if SETTINGS.enable_redoc else "https://stdapi.ai/api_reference/")
    )
}

_mcp_streamable = SETTINGS.enable_mcp_streamable_http
_mcp_sse = SETTINGS.enable_mcp_sse

#: Link header value for agent discovery (RFC 8288), None if no resources available
_LINK_HEADER: str | None = (
    ", ".join(
        part
        for part in [
            (
                '</openapi.json>; rel="service-desc"'
                if SETTINGS.enable_openapi_json
                else None
            ),
            (
                '</docs>; rel="service-doc"'
                if SETTINGS.enable_docs
                else ('</redoc>; rel="service-doc"' if SETTINGS.enable_redoc else None)
            ),
            (
                '</.well-known/mcp/server-card.json>; rel="mcp-server-card"'
                if _mcp_streamable or _mcp_sse
                else None
            ),
        ]
        if part is not None
    )
    or None
)

#: RFC 9727 / RFC 9264 Linkset API catalog for agent discovery.
_API_CATALOG: dict[str, Any] = {
    "linkset": [
        {
            "anchor": "/.well-known/api-catalog",
            **(
                {
                    "service-desc": [
                        {
                            "href": "/openapi.json",
                            "type": "application/vnd.oai.openapi+json;version=3.0",
                        }
                    ]
                }
                if SETTINGS.enable_openapi_json
                else {}
            ),
            **(
                {"service-doc": [{"href": "/docs", "title": "Swagger UI"}]}
                if SETTINGS.enable_docs
                else (
                    {"service-doc": [{"href": "/redoc", "title": "ReDoc"}]}
                    if SETTINGS.enable_redoc
                    else {}
                )
            ),
            **(
                {"mcp-server-card": [{"href": "/.well-known/mcp/server-card.json"}]}
                if _mcp_streamable or _mcp_sse
                else {}
            ),
        }
    ]
}


#: MCP Server Card (SEP-1649) — served when at least one MCP transport is enabled.
MCP_SERVER_CARD: dict[str, Any] | None = (
    {
        "$schema": "https://static.modelcontextprotocol.io/schemas/mcp-server-card/v1.json",
        "version": "1.0",
        "protocolVersion": "2025-03-26",
        "serverInfo": {"name": EDITION_TITLE, "version": SERVER_VERSION},
        "transport": (
            {"type": "streamable-http", "endpoint": "/mcp"}
            if _mcp_streamable
            else {"type": "sse", "endpoint": "/sse"}
        ),
        "capabilities": {"tools": {}},
    }
    if _mcp_streamable or _mcp_sse
    else None
)


#: Robots.txt content - allows docs/openapi/.well-known, disallows everything else
_ROBOTS_TXT = "\n".join(
    part
    for part in (
        "User-agent: *",
        "Content-Signal: ai-train=yes, search=yes, ai-input=yes",
        "Allow: /docs" if SETTINGS.enable_docs else None,
        "Allow: /redoc" if SETTINGS.enable_redoc else None,
        "Allow: /openapi.json" if SETTINGS.enable_openapi_json else None,
        "Allow: /.well-known/",
        "Disallow: /",
    )
    if part is not None
)


#: Pre-rendered response for the root endpoint, identical on every request.
_ROOT_RESPONSE = JSONResponse(_WELCOME)
if _LINK_HEADER:
    _ROOT_RESPONSE.headers["Link"] = _LINK_HEADER

#: Pre-rendered response for the API catalog endpoint, identical on every request.
_API_CATALOG_RESPONSE = JSONResponse(
    _API_CATALOG, media_type="application/linkset+json"
)


@router.get("/")
async def root() -> JSONResponse:
    """Return a welcome message for the API root endpoint.

    Returns:
        JSONResponse containing a welcome message with Link headers for agent discovery.
    """
    return _ROOT_RESPONSE


@router.get("/.well-known/api-catalog")
async def api_catalog() -> JSONResponse:
    """RFC 9727 API catalog for agent discovery.

    Returns a Linkset document (RFC 9264) advertising the API's OpenAPI schema,
    documentation, and MCP server card to enable automated agent discovery.

    Returns:
        Linkset document with Content-Type application/linkset+json.
    """
    return _API_CATALOG_RESPONSE


@router.get("/.well-known/mcp/server-card.json")
async def mcp_server_card() -> JSONResponse:
    """MCP Server Card (SEP-1649) for agent discovery.

    Advertises available MCP transports and capabilities to AI agents.
    Only active when at least one MCP transport is enabled.

    Returns:
        MCP Server Card document, or 404 if no MCP transport is enabled.
    """
    if MCP_SERVER_CARD is None:
        return JSONResponse({"error": "MCP is not enabled"}, status_code=404)
    return JSONResponse(MCP_SERVER_CARD)


@router.get("/robots.txt")
async def robots_txt() -> PlainTextResponse:
    """Robots.txt optimized for API discovery by AI agents."""
    return PlainTextResponse(_ROBOTS_TXT)


#: Pre-rendered response for the health check endpoint, identical on every request.
_HEALTH_RESPONSE = JSONResponse(asdict(HealthResponse()))

#: Pre-rendered response for the readiness probe endpoint, identical on every request.
_PING_RESPONSE = JSONResponse(asdict(PingResponse()))


@router.get("/health")
async def health_check() -> JSONResponse:
    """Check if the service is healthy and operational.

    Returns:
        JSONResponse with status "ok" when the service is operational
    """
    return _HEALTH_RESPONSE


@router.get("/ping")
async def ping() -> JSONResponse:
    """Report readiness in the shape Amazon Bedrock AgentCore Runtime expects.

    Returns:
        JSONResponse with status "Healthy" when the service is operational.
    """
    return _PING_RESPONSE
