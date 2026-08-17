"""Every path this server advertises follows the configured routes prefix.

A deployment that sets ``openai_routes_prefix`` moves the whole OpenAI-compatible
surface under it, so anything the server hands a client -- the WebSocket URL a
realtime client secret is spent on, a filter example an MCP agent copies, an
error message naming the endpoint to poll -- has to move with it. A path written
as a literal points at nothing, and a client has no way to guess that the address
it was given is the wrong one.

Route payloads are built at import time, so each test re-executes the module
under a non-default prefix as a private copy -- never a reload, which would
replace the running application's own payloads.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/164
     stdapi/config.py:Settings.openai_routes_prefix
"""

from __future__ import annotations

import re
from importlib.util import find_spec, module_from_spec
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI

from stdapi.config import SETTINGS
from stdapi.models.capabilities import ROUTE_CAPABILITIES
from stdapi.utils import to_json_str

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

pytestmark = pytest.mark.local

#: Non-default prefix the modules under test are re-executed with.
_PREFIX = "/openai"

#: A dialect path as it appears inside a user-visible string.
_V1_PATH = re.compile(r"/v1/[\w{}./-]*")


def _unprefixed(text: str) -> list[str]:
    """Return every dialect path in *text* that does not carry ``_PREFIX``.

    Args:
        text: User-visible text to scan.

    Returns:
        The offending paths, empty when every one of them is prefixed.
    """
    return [
        match.group()
        for match in _V1_PATH.finditer(text)
        if not text[: match.start()].endswith(_PREFIX)
    ]


def _advertised_text(module: ModuleType) -> str:
    """Collect what *module*'s routes tell a caller, mounted paths included.

    Only the ``paths`` section is read: it holds the mounted addresses, the route
    descriptions and the request-parameter descriptions, which together are what
    an MCP client reads before it builds a call. Response schema prose describes
    the API rather than addressing it, and stays out.

    Args:
        module: A re-executed route module exposing a ``router``.

    Returns:
        The mounted paths and every description they carry, as one text.
    """
    app = FastAPI()
    app.include_router(module.router)
    paths = app.openapi()["paths"]
    return to_json_str(sorted(paths)) + to_json_str(
        [value for path in paths.values() for value in _descriptions(path)]
    )


def _descriptions(value: object) -> Iterator[str]:
    """Yield every ``summary`` and ``description`` string nested in *value*.

    Args:
        value: A fragment of an OpenAPI document.

    Yields:
        Each description or summary found, at any depth.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"summary", "description"} and isinstance(nested, str):
                yield nested
            yield from _descriptions(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _descriptions(nested)


@pytest.fixture
def prefixed_module(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], ModuleType]]:
    """Build private copies of route modules under a non-default routes prefix.

    Re-executing a route module registers its route capabilities again, so the
    registry is snapshotted and restored rather than left holding prefixed paths
    for the rest of the session.

    Yields:
        A callable turning a module name into its re-executed private copy.
    """
    saved = dict(ROUTE_CAPABILITIES)
    monkeypatch.setattr(SETTINGS, "openai_routes_prefix", _PREFIX)

    def build(name: str) -> ModuleType:
        spec = find_spec(name)
        assert spec is not None
        assert spec.loader is not None
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield build
    ROUTE_CAPABILITIES.clear()
    ROUTE_CAPABILITIES.update(saved)


def test_realtime_advertises_the_websocket_it_is_mounted_at(
    prefixed_module: Callable[[str], ModuleType],
) -> None:
    """The client secret route advertises the prefixed realtime WebSocket path.

    The secret is minted for an untrusted client that has nothing but this
    description to dial from: an unprefixed address is a 404 on connect with
    nothing to suggest the advertised path is the wrong one.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/164
         stdapi/routes/openai_realtime.py:create_realtime_client_secret
    """
    module = prefixed_module("stdapi.routes.openai_realtime")
    document = _advertised_text(module)
    assert module.router.prefix == f"{_PREFIX}/v1/realtime"
    assert f"wss://<host>{_PREFIX}/v1/realtime" in document
    assert not _unprefixed(document)


def test_search_models_route_examples_follow_the_prefix(
    prefixed_module: Callable[[str], ModuleType],
) -> None:
    """The ``route`` filter examples name paths this deployment actually serves.

    ``search_models`` indexes the mounted path, so an agent copying an example
    written as a literal gets a 400 telling it no model supports that route.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/164
         stdapi/routes/core_models.py:search_models
    """
    module = prefixed_module("stdapi.routes.core_models")
    document = _advertised_text(module)
    assert f"route={_PREFIX}/v1/images/generations" in document
    assert not _unprefixed(document)


def test_unfinished_video_names_the_status_route_it_serves(
    prefixed_module: Callable[[str], ModuleType],
) -> None:
    """The 404 on an unfinished video points at the prefixed retrieval route.

    The message exists to tell a client where to poll, so a literal path sends
    it to an address this deployment does not answer on.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/164
         stdapi/routes/openai_videos.py:get_video_content
    """
    module = prefixed_module("stdapi.routes.openai_videos")
    message = module._VIDEO_NOT_READY_MESSAGE  # noqa: SLF001
    assert f"{_PREFIX}/v1/videos/" in message
    assert not _unprefixed(message)
    assert not _unprefixed(_advertised_text(module))
