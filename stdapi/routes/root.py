"""Root endpoint for the API."""

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from stdapi.config import SETTINGS
from stdapi.types import ApiCatalogLink

if TYPE_CHECKING:
    from stdapi.types import ApiCatalog

router = APIRouter()

#: Welcome message payload for root endpoint
_WELCOME = {
    "message": "Welcome to the stdapi.ai API! Documentation is available at "
    + (
        "/docs"
        if SETTINGS.enable_docs
        else ("/redoc" if SETTINGS.enable_redoc else "https://stdapi.ai/api_reference/")
    )
}

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
        ]
        if part is not None
    )
    or None
)

#: RFC 9727 API catalog response for agent discovery
_API_CATALOG: ApiCatalog = {
    "api_version": "1.0.0",
    "description": "stdapi.ai - OpenAI-compatible API gateway for AWS Bedrock",
    "links": [
        *(
            [
                ApiCatalogLink(
                    rel="service-desc", href="/openapi.json", title="OpenAPI Schema"
                )
            ]
            if SETTINGS.enable_openapi_json
            else ()
        ),
        *(
            [ApiCatalogLink(rel="service-doc", href="/docs", title="Swagger UI")]
            if SETTINGS.enable_docs
            else (
                [ApiCatalogLink(rel="service-doc", href="/redoc", title="ReDoc")]
                if SETTINGS.enable_redoc
                else ()
            )
        ),
    ],
}


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


@router.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Return a welcome message for the API root endpoint.

    Returns:
        JSONResponse containing a welcome message with Link headers for agent discovery.
    """
    response = JSONResponse(_WELCOME)
    if _LINK_HEADER:
        response.headers["Link"] = _LINK_HEADER
    return response


@router.get("/.well-known/api-catalog", include_in_schema=False, tags=["metadata"])
async def api_catalog() -> JSONResponse:
    """RFC 9727 API catalog for agent discovery.

    Returns a JSON document containing links to the API OpenAPI schema
    and documentation, enabling AI agents to discover available resources.

    Returns:
        API catalog.
    """
    return JSONResponse(_API_CATALOG)


@router.get("/robots.txt", include_in_schema=False, tags=["metadata"])
async def robots_txt() -> PlainTextResponse:
    """Robots.txt optimized for API discovery by AI agents."""
    return PlainTextResponse(_ROBOTS_TXT)
