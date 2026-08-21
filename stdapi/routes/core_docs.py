"""Interactive API documentation pages, and everything a browser loads with them.

FastAPI's built-in pages reach three third parties: an icon on
``fastapi.tiangolo.com``, Swagger UI and ReDoc on ``cdn.jsdelivr.net`` at
floating major tags, and a ReDoc web font on ``fonts.googleapis.com``. An
air-gapped or egress-restricted deployment therefore gets a blank page, and
every other deployment runs whatever those tags resolve to at that moment.

These replacements serve the gateway's own mark, and the pinned scripts the
image build fetched, from the gateway itself. A source checkout has fetched
nothing, so the pages fall back to the publisher URL of the same exact
version -- never a floating tag.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/184
     https://github.com/stdapi-ai/stdapi.ai/issues/185
     stdapi/docs_assets/__init__.py
"""

from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse, Response

from stdapi.config import SETTINGS
from stdapi.docs_assets import ASSETS_PATH, BROWSER_ASSETS, LOCAL_ASSETS

if TYPE_CHECKING:
    from pathlib import Path

router = APIRouter(tags=["metadata"], include_in_schema=False)

#: Address the gateway serves its own icon at, also the one browsers probe unprompted.
FAVICON_PATH = "/favicon.ico"

#: Address Swagger UI returns an OAuth 2.0 authorization code to, as FastAPI mounts it.
_OAUTH2_REDIRECT_PATH = "/docs/oauth2-redirect"

#: Served name of the Swagger UI script bundle.
_SWAGGER_UI_JS = "swagger-ui-bundle.js"

#: Served name of the Swagger UI stylesheet.
_SWAGGER_UI_CSS = "swagger-ui.css"

#: Served name of the ReDoc standalone script bundle.
_REDOC_JS = "redoc.standalone.js"

#: Seconds a client may reuse a static file, which only changes with the server version.
_STATIC_MAX_AGE = 86400

#: The gateway mark, read once at import; the icon the documentation site also uses.
_FAVICON = (files("stdapi") / "favicon.svg").read_bytes()

#: Headers a static file is served with, copied into a fresh response per request.
_STATIC_HEADERS = {"cache-control": f"public, max-age={_STATIC_MAX_AGE}"}


def _mounted(request: Request, path: str) -> str:
    """Return *path* as the client has to address it.

    A deployment behind a proxy or mounted under a prefix serves the whole
    application below its ASGI root path, so a page-relative address written
    without it points at nothing.

    Args:
        request: Incoming request, carrying the ASGI root path.
        path: Application-relative path.

    Returns:
        The address to write into the page.
    """
    return f"{request.scope.get('root_path', '').rstrip('/')}{path}"


def _asset_url(request: Request, name: str) -> str:
    """Return the address the page loads the asset named *name* from.

    The fetched copy wins whenever the build placed one beside the package,
    which is what makes the page render with no egress at all. Without one --
    a source checkout, and only a source checkout -- the page names the
    publisher, still at the pinned exact version.

    Args:
        request: Incoming request, carrying the ASGI root path.
        name: Served name of the asset.

    Returns:
        The address to write into the page.
    """
    if name in LOCAL_ASSETS:
        return _mounted(request, f"{ASSETS_PATH}/{name}")
    return BROWSER_ASSETS[name].url


@cache
def _asset_body(path: Path) -> bytes:
    """Return the bytes of a fetched asset, read from disk once.

    Keyed on the file rather than on its served name: a megabyte and a half is
    read on the first request for it and referenced by every later one.

    Args:
        path: The fetched file.

    Returns:
        Its bytes, shared by every response that serves it.
    """
    return path.read_bytes()


@router.get(FAVICON_PATH)
async def favicon() -> Response:
    """Serve the icon the API documentation pages are branded with.

    A response is built per request rather than shared: the compression
    middleware rewrites the headers of whatever object it is handed, and a
    reused one would carry them into the next, uncompressed, request.

    Returns:
        The gateway mark, as a cacheable SVG document.
    """
    return Response(_FAVICON, media_type="image/svg+xml", headers=_STATIC_HEADERS)


@router.get(f"{ASSETS_PATH}/{{name}}")
async def docs_asset(name: str) -> Response:
    """Serve one of the pinned scripts the documentation pages load.

    Registered whatever the settings say, like the icon: the pages are what
    decide whether anything ever asks for these, and a route that appeared only
    with them would answer nothing to a browser holding a cached page.

    Args:
        name: File name, which must be one the image build fetched.

    Returns:
        The asset, as a cacheable document of its own type.

    Raises:
        HTTPException: 404 when this deployment fetched no such file, which is
            every one of them in a source checkout -- where the pages name the
            publisher instead and nothing requests this route.
    """
    if (path := LOCAL_ASSETS.get(name)) is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return Response(
        _asset_body(path),
        media_type=BROWSER_ASSETS[name].media_type,
        headers=_STATIC_HEADERS,
    )


if SETTINGS.enable_docs:

    @router.get("/docs")
    async def swagger_ui_html(request: Request) -> HTMLResponse:
        """Serve the Swagger UI documentation page.

        Args:
            request: Incoming request, read for the deployment's root path.

        Returns:
            The Swagger UI page, branded with the gateway's own icon and
            loading the pinned Swagger UI this deployment fetched.
        """
        return get_swagger_ui_html(
            openapi_url=_mounted(request, request.app.openapi_url),
            title=f"{request.app.title} - Swagger UI",
            oauth2_redirect_url=_mounted(request, _OAUTH2_REDIRECT_PATH),
            swagger_favicon_url=_mounted(request, FAVICON_PATH),
            swagger_js_url=_asset_url(request, _SWAGGER_UI_JS),
            swagger_css_url=_asset_url(request, _SWAGGER_UI_CSS),
        )

    @router.get(_OAUTH2_REDIRECT_PATH)
    async def swagger_ui_redirect() -> HTMLResponse:
        """Hand an OAuth 2.0 authorization code back to the Swagger UI page.

        Returns:
            The redirect page Swagger UI's authorization flow ends on.
        """
        return get_swagger_ui_oauth2_redirect_html()


if SETTINGS.enable_redoc:

    @router.get("/redoc")
    async def redoc_html(request: Request) -> HTMLResponse:
        """Serve the ReDoc documentation page.

        Args:
            request: Incoming request, read for the deployment's root path.

        Returns:
            The ReDoc page, branded with the gateway's own icon and loading the
            pinned ReDoc this deployment fetched.
        """
        return get_redoc_html(
            openapi_url=_mounted(request, request.app.openapi_url),
            title=f"{request.app.title} - ReDoc",
            redoc_favicon_url=_mounted(request, FAVICON_PATH),
            redoc_js_url=_asset_url(request, _REDOC_JS),
            # FastAPI otherwise links a Google-hosted web font into the page,
            # which is a third party this page must not need either. Without it
            # ReDoc renders in the reader's own sans-serif.
            with_google_fonts=False,
        )
