"""Ollama-compatible base URL probe.

An Ollama server answers ``GET /`` and ``HEAD /`` with the plain-text body
``Ollama is running``, and a client's "test connection" action probes the base
URL it was configured with rather than an API endpoint. That base URL is this
dialect's routes prefix, so the probe is served there.

- GET  {prefix} — report that the Ollama-compatible surface is serving
- HEAD {prefix} — the same, without a body
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.responses import PlainTextResponse, Response

from stdapi.api_providers.ollama import TAG_OLLAMA
from stdapi.auth import authenticate
from stdapi.config import SETTINGS

#: Body an Ollama server answers its base URL with.
_RUNNING: str = "Ollama is running"

router: APIRouter | None

if SETTINGS.ollama_routes_prefix:
    #: The base URL an operator points a client at, and its slashed spelling.
    _BASE_PATHS: tuple[str, str] = (
        SETTINGS.ollama_routes_prefix,
        f"{SETTINGS.ollama_routes_prefix}/",
    )

    _router = APIRouter(tags=["Models", TAG_OLLAMA])

    # Two paths rather than one: an ``APIRoute`` does not add HEAD to a GET, and
    # a probe whose HTTP client does not follow redirects fails on the 307 the
    # slash would otherwise earn. Kept out of the schema, like the other Ollama
    # probes: it publishes nothing an MCP tool could carry.
    @_router.get(_BASE_PATHS[0], include_in_schema=False)
    @_router.get(_BASE_PATHS[1], include_in_schema=False)
    async def root(
        _: Annotated[None, Depends(authenticate)] = None,
    ) -> PlainTextResponse:
        """Report that the Ollama-compatible surface is serving.

        Returns:
            The plain-text body an Ollama server answers its base URL with.
        """
        return PlainTextResponse(_RUNNING)

    @_router.head(_BASE_PATHS[0], include_in_schema=False)
    @_router.head(_BASE_PATHS[1], include_in_schema=False)
    async def head_root(_: Annotated[None, Depends(authenticate)] = None) -> Response:
        """Answer a liveness probe on the base URL.

        Returns:
            An empty 200 response.
        """
        return Response(status_code=200)

    router = _router

else:
    # Mounted at the root: "/" is the server's own welcome document, which owns
    # the discovery Link header and the RFC 9727 catalog. The probe is not served.
    router = None
