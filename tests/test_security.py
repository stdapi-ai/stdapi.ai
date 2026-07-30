"""SSRF validation in :mod:`stdapi.security`.

Two layers are covered: :func:`stdapi.security.validate_host_ssrf`, which
classifies a host, and the aiohttp connector returned by
:func:`stdapi.security.ssrf_safe_connector`, which enforces the classification at
connect time so every redirect hop is re-checked and the connection is pinned to
the address that was validated.

403 means "policy block" and 400 means "unusable URL"; the distinction is what
``ssrf_blocked_status`` reports to the client, so every rejection test asserts the
status *and* the message rather than just the exception class.

Ref: stdapi/security.py:validate_host_ssrf
     stdapi/security.py:ssrf_safe_connector
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aiohttp import ClientConnectorError, ClientSession, TCPConnector, web
from aiohttp.test_utils import TestServer

from stdapi import security
from stdapi.api_errors import ApiError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

pytestmark = pytest.mark.local


@pytest.fixture
async def start_test_server() -> AsyncGenerator[
    Callable[[web.Application], Awaitable[TestServer]]
]:
    """Start aiohttp test servers, closing every one of them on teardown.

    Yields:
        A coroutine function starting a server for the given application.
    """
    servers: list[TestServer] = []

    async def _start(app: web.Application) -> TestServer:
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return server

    yield _start
    for server in servers:
        await server.close()


def _forbid_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any call to :func:`stdapi.security._resolve_hostname` fail the test."""

    async def _fail(hostname: str) -> list[str]:
        msg = f"DNS resolution must not be attempted for {hostname!r}"
        raise AssertionError(msg)

    monkeypatch.setattr(security, "_resolve_hostname", _fail)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.0.0.5",
        "192.168.1.1",
        "::1",
        "0.0.0.0",  # noqa: S104
    ],
)
async def test_validate_host_ssrf_blocks_unsafe_ip_literals(
    host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe IP-literal hosts are rejected as 403 without any DNS resolution.

    The list covers each rejected class: loopback (v4 and v6), the EC2 metadata
    link-local address, RFC 1918 private ranges and the unspecified address.
    Resolution is sabotaged so taking the literal fast path is part of the claim.
    """
    _forbid_resolution(monkeypatch)
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf(host)
    assert exc.value.status == 403
    assert str(exc.value) == f"Forbidden host in URL: {host}."


async def test_validate_host_ssrf_allows_public_ip_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public IP-literal host passes validation and pins to itself.

    The returned list is what the connector will dial, so a literal must come
    back unchanged rather than being re-resolved.
    """
    _forbid_resolution(monkeypatch)
    assert await security.validate_host_ssrf("93.184.216.34") == ["93.184.216.34"]


def _patch_resolution(monkeypatch: pytest.MonkeyPatch, addresses: list[str]) -> None:
    """Force :func:`stdapi.security._resolve_hostname` to return *addresses*."""

    async def _fake(_: str) -> list[str]:
        return addresses

    monkeypatch.setattr(security, "_resolve_hostname", _fake)


async def test_validate_host_ssrf_blocks_hostname_resolving_to_unsafe_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named host resolving to a loopback/private address is rejected as 403.

    The message names the hostname, not the resolved address, so the block cannot
    be used to enumerate internal addresses.
    """
    _patch_resolution(monkeypatch, ["127.0.0.1"])
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf("example.test")
    assert exc.value.status == 403
    assert str(exc.value) == "Forbidden host in URL: example.test."
    assert "127.0.0.1" not in str(exc.value)


async def test_validate_host_ssrf_allows_hostname_resolving_to_public_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named host resolving only to public addresses passes validation.

    The resolved addresses are returned so the caller connects to exactly what
    was checked, closing the DNS-rebinding window.
    """
    _patch_resolution(monkeypatch, ["93.184.216.34"])
    assert await security.validate_host_ssrf("example.test") == ["93.184.216.34"]


async def test_validate_host_ssrf_fails_closed_on_unresolvable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that resolves to no address is rejected as a bad URL (400).

    An empty address list must never be read as "nothing unsafe was found": it is
    reported as an unusable URL, distinct from the 403 policy block.
    """
    _patch_resolution(monkeypatch, [])
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf("nonexistent.test")
    assert exc.value.status == 400
    assert str(exc.value) == "Cannot resolve host in URL: nonexistent.test."


async def test_validate_host_ssrf_rejects_unparseable_resolved_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolved address that cannot be parsed is treated as unsafe (fail closed).

    ``_is_unsafe_ip`` returns True on a ``ValueError`` from ``ip_address``, so a
    malformed resolver answer becomes a 403 rather than being waved through.

    Ref: stdapi/security.py:_is_unsafe_ip
    """
    _patch_resolution(monkeypatch, ["not-an-ip"])
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf("example.test")
    assert exc.value.status == 403
    assert str(exc.value) == "Forbidden host in URL: example.test."


async def test_ssrf_safe_connector_returns_validating_connector() -> None:
    """Each call returns a fresh connector wired to the one shared SSRF resolver.

    The connector is owned (and closed) by its ``ClientSession``, so it cannot be
    shared; the resolver is stateless and must be, otherwise every session would
    build its own DNS resolver.
    """
    connector = security.ssrf_safe_connector()
    other = security.ssrf_safe_connector()
    try:
        assert isinstance(connector, TCPConnector)
        assert connector is not other
        assert connector._resolver is security._SHARED_RESOLVER  # noqa: SLF001
        assert other._resolver is security._SHARED_RESOLVER  # noqa: SLF001
        assert isinstance(security._SHARED_RESOLVER, security._SsrfSafeResolver)  # noqa: SLF001
    finally:
        await connector.close()
        await other.close()


class TestSsrfSafeConnectorIntegration:
    """End-to-end SSRF enforcement through the aiohttp connector path.

    Ref: stdapi/security.py:_SsrfSafeConnector
         stdapi/security.py:ssrf_blocked_status
    """

    async def test_unsafe_ip_literal_blocked_before_connecting(self) -> None:
        """A request to an unsafe IP literal is rejected without connecting.

        aiohttp bypasses the configured resolver for IP literals, so the
        connector subclass has to validate them itself; the rejection surfaces as
        an ``SsrfBlockedError`` wrapped in ``ClientConnectorError`` and maps to 403.
        """
        async with ClientSession(connector=security.ssrf_safe_connector()) as session:
            with pytest.raises(ClientConnectorError) as exc:
                await session.get("http://169.254.169.254/")
        assert isinstance(exc.value.os_error, security.SsrfBlockedError)
        assert "169.254.169.254" in str(exc.value.os_error)
        assert security.ssrf_blocked_status(exc.value) == 403

    async def test_hostname_resolving_to_unsafe_ip_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A named host resolving to an unsafe address is rejected at connect time.

        Validation happens in the resolver, so the TCP connection is never
        attempted and the failure is reported as a 403 policy block.
        """
        _patch_resolution(monkeypatch, ["169.254.169.254"])
        async with ClientSession(connector=security.ssrf_safe_connector()) as session:
            with pytest.raises(ClientConnectorError) as exc:
                await session.get("http://internal.test/")
        assert isinstance(exc.value.os_error, security.SsrfBlockedError)
        assert "internal.test" in str(exc.value.os_error)
        assert security.ssrf_blocked_status(exc.value) == 403

    async def test_unresolvable_hostname_maps_to_plain_fetch_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolvable host surfaces as a DNS failure (400), not an SSRF block.

        The resolver raises a plain ``OSError`` for the 400 case, so a typo'd URL
        is not reported to the caller as a blocked target.
        """
        _patch_resolution(monkeypatch, [])
        async with ClientSession(connector=security.ssrf_safe_connector()) as session:
            with pytest.raises(ClientConnectorError) as exc:
                await session.get("http://unresolvable.test/")
        assert not isinstance(exc.value.os_error, security.SsrfBlockedError)
        assert security.ssrf_blocked_status(exc.value) == 400

    async def test_redirect_hop_to_unsafe_target_blocked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        start_test_server: Callable[[web.Application], Awaitable[TestServer]],
    ) -> None:
        """A redirect from a safe host to an unsafe IP-literal target is blocked at the hop.

        The first hop must succeed for the test to be meaningful, so the local
        test server's loopback address is temporarily treated as safe while every
        other address stays unsafe.
        """
        monkeypatch.setattr(
            security, "_is_unsafe_ip", lambda ip: str(ip) != "127.0.0.1"
        )

        async def redirect(_: web.Request) -> web.Response:
            raise web.HTTPFound(location="http://169.254.169.254/")

        app = web.Application()
        app.router.add_get("/redirect", redirect)
        server = await start_test_server(app)

        async with ClientSession(connector=security.ssrf_safe_connector()) as session:
            with pytest.raises(ClientConnectorError) as exc:
                await session.get(server.make_url("/redirect"))
        assert isinstance(exc.value.os_error, security.SsrfBlockedError)
        assert "169.254.169.254" in str(exc.value.os_error)
        assert security.ssrf_blocked_status(exc.value) == 403

    async def test_connection_pinned_to_validated_address(
        self,
        monkeypatch: pytest.MonkeyPatch,
        start_test_server: Callable[[web.Application], Awaitable[TestServer]],
    ) -> None:
        """A named host connects to the resolver-validated address, not system DNS.

        ``pinned.test`` does not exist in DNS, so a successful response can only
        come from the address the SSRF resolver returned.
        """
        monkeypatch.setattr(
            security, "_is_unsafe_ip", lambda ip: str(ip) != "127.0.0.1"
        )

        async def ok(_: web.Request) -> web.Response:
            return web.Response(text="pinned")

        app = web.Application()
        app.router.add_get("/ok", ok)
        server = await start_test_server(app)

        _patch_resolution(monkeypatch, ["127.0.0.1"])
        async with (
            ClientSession(connector=security.ssrf_safe_connector()) as session,
            session.get(f"http://pinned.test:{server.port}/ok") as resp,
        ):
            assert resp.status == 200
            assert await resp.text() == "pinned"
