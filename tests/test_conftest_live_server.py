"""The loopback server backing the WebSocket lane, and the loop it serves on.

``live_server`` serves the same ASGI app as ``test_client``, from a real uvicorn
on a loopback port, and runs it in that client's own portal with uvicorn's
lifespan off. The process then keeps a single app lifespan, and with it a single
AWS client pool: two lifespans share that process-wide pool, so requests served
by one loop reach clients the other owns, and the aiohttp connector opened on
first use then belongs to a loop that is not the one closing it.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/271
     tests/conftest.py:live_server
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

from stdapi.api_providers import set_response_headers

if TYPE_CHECKING:
    import pytest
    from fastapi import Request, Response
    from starlette.testclient import TestClient

#: Seconds the loopback server gets to answer the probe request.
_PROBE_TIMEOUT = 30.0


async def _running_loop() -> asyncio.AbstractEventLoop:
    """Return the event loop the call runs on.

    Returns:
        The loop of the portal that ran this coroutine.
    """
    return asyncio.get_running_loop()


class TestLiveServerLoop:
    """The loopback server and the ASGI test client share one event loop."""

    def test_the_live_server_serves_on_the_test_client_loop(
        self,
        live_server: str | None,
        local_test_client: TestClient,
        api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A request dialled over the socket is handled on the test client's loop.

        Serving from a loop of its own instead leaves the AWS client pool shared
        between two loops: the pool then hands the serving loop clients another
        lifespan owns, and closing one of those raises ``RuntimeError: loop ...
        is not the running loop`` when the session unwinds.
        """
        portal = local_test_client.portal
        assert portal is not None, "the test client holds its portal while entered"
        served_on: list[asyncio.AbstractEventLoop] = []

        def recording(request: Request, response: Response, processing_ms: int) -> None:
            """Record the loop serving the request, then set the headers."""
            served_on.append(asyncio.get_running_loop())
            set_response_headers(request, response, processing_ms)

        # The app's own middleware calls it on every logged request, on the loop
        # serving that request.
        monkeypatch.setattr("stdapi.main.set_response_headers", recording)
        response = httpx.get(
            f"{live_server}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_PROBE_TIMEOUT,
        )

        assert response.status_code == 200, response.text
        assert served_on, "the request never reached the request-logging middleware"
        assert served_on[0] is portal.call(_running_loop)
