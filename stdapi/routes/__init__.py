"""FastAPI routers auto-discovery for the routes package.

Only the modules' ``router`` is discovered: anything a route needs at startup
is wired separately, in the lifespan in stdapi/main.py.
"""

from importlib import import_module
from pkgutil import iter_modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def discover_routers(app: FastAPI) -> None:
    """Discover submodules and include their ``router`` into the FastAPI app.

    A module may set ``router = None`` to opt out of registration (e.g. when its
    routes would collide with another provider's at the same prefix).

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
            router = module.router
        except AttributeError as exc:  # pragma: no cover
            msg = f"Module {__name__}.{name} has an invalid 'router'"
            raise ImportError(msg) from exc
        if router is None:
            continue
        try:
            app.include_router(router)
        except TypeError as exc:  # pragma: no cover
            msg = f"Module {__name__}.{name} has an invalid 'router'"
            raise ImportError(msg) from exc
