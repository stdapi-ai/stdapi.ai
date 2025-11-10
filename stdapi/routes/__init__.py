"""FastAPI routers auto-discovery for the routes package.

This package initializer discovers submodules in stdapi.routes, imports them,
and collects any top-level variable named "router" that is an instance of
fastapi.APIRouter. The discovered routers are exposed via the ROUTERS list
and the list_routers() helper.

Design:
- On import, iterate over sibling modules and import them if not private.
- If a module defines a top-level variable named "router" which is an
  APIRouter, add it to the ROUTERS registry.
- Idempotent: importing the package multiple times does not duplicate entries.

Note:
- Startup functions (like initialize_voices in audio_speech) are not handled
  by this mechanism and should be wired where needed (e.g., in lifespan).
"""

from importlib import import_module
from pkgutil import iter_modules

from fastapi import FastAPI


def discover_routers(app: FastAPI) -> None:
    """Discover submodules and include their routers into the FastAPI app.

    Iterates over stdapi.routes submodules, imports each non-private module, and
    includes its top-level variable named "router" into the provided FastAPI
    application.

    Args:
        app: The FastAPI application into which discovered routers are included.

    Raises:
        ImportError: If a discovered module does not expose a top-level
            variable named "router", or if the value is not a compatible
            FastAPI router. The underlying AttributeError or TypeError is
            attached as the cause.
    """
    for module_info in iter_modules(import_module(__name__).__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        module = import_module(f"{__name__}.{name}")
        try:
            app.include_router(module.router)
        except (TypeError, AttributeError) as exc:  # pragma: no cover
            msg = f"Module {__name__}.{name} has an invalid 'router'"
            raise ImportError(msg) from exc
