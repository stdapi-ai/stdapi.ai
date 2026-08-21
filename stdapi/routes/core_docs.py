"""Interactive API documentation pages and the icon they are branded with.

FastAPI's built-in pages point at an icon hosted on ``fastapi.tiangolo.com``, so
every browser opening them reaches a third party. These replacements serve the
gateway's own mark from the gateway itself, which is what an air-gapped or
egress-restricted deployment needs and what keeps the request on-host.
"""

from importlib.resources import files

from fastapi import APIRouter, Request
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse, Response

from stdapi.config import SETTINGS

router = APIRouter(tags=["metadata"], include_in_schema=False)

#: Address the gateway serves its own icon at, also the one browsers probe unprompted.
FAVICON_PATH = "/favicon.ico"

#: Address Swagger UI returns an OAuth 2.0 authorization code to, as FastAPI mounts it.
_OAUTH2_REDIRECT_PATH = "/docs/oauth2-redirect"

#: Seconds a client may reuse the icon, which only changes with the server version.
_FAVICON_MAX_AGE = 86400

#: The gateway mark, read once at import; the icon the documentation site also uses.
_FAVICON = (files("stdapi") / "favicon.svg").read_bytes()

#: Headers the icon is served with, copied into a fresh response on every request.
_FAVICON_HEADERS = {"cache-control": f"public, max-age={_FAVICON_MAX_AGE}"}


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


@router.get(FAVICON_PATH)
async def favicon() -> Response:
    """Serve the icon the API documentation pages are branded with.

    A response is built per request rather than shared: the compression
    middleware rewrites the headers of whatever object it is handed, and a
    reused one would carry them into the next, uncompressed, request.

    Returns:
        The gateway mark, as a cacheable SVG document.
    """
    return Response(_FAVICON, media_type="image/svg+xml", headers=_FAVICON_HEADERS)


if SETTINGS.enable_docs:

    @router.get("/docs")
    async def swagger_ui_html(request: Request) -> HTMLResponse:
        """Serve the Swagger UI documentation page.

        Args:
            request: Incoming request, read for the deployment's root path.

        Returns:
            The Swagger UI page, branded with the gateway's own icon.
        """
        return get_swagger_ui_html(
            openapi_url=_mounted(request, request.app.openapi_url),
            title=f"{request.app.title} - Swagger UI",
            oauth2_redirect_url=_mounted(request, _OAUTH2_REDIRECT_PATH),
            swagger_favicon_url=_mounted(request, FAVICON_PATH),
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
            The ReDoc page, branded with the gateway's own icon.
        """
        return get_redoc_html(
            openapi_url=_mounted(request, request.app.openapi_url),
            title=f"{request.app.title} - ReDoc",
            redoc_favicon_url=_mounted(request, FAVICON_PATH),
        )
