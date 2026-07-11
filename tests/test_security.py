"""Unit tests for SSRF validation (:mod:`stdapi.security`)."""

from __future__ import annotations

import pytest
from aiohttp import ClientConnectorError, ClientSession, TCPConnector, web
from aiohttp.test_utils import TestServer

from stdapi import security
from stdapi.api_errors import ApiError


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
async def test_validate_host_ssrf_blocks_unsafe_ip_literals(host: str) -> None:
    """Unsafe IP-literal hosts are rejected without any DNS resolution."""
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf(host)
    assert exc.value.status == 403


async def test_validate_host_ssrf_allows_public_ip_literal() -> None:
    """A public IP-literal host passes validation and pins to itself."""
    assert await security.validate_host_ssrf("93.184.216.34") == ["93.184.216.34"]


def _patch_resolution(monkeypatch: pytest.MonkeyPatch, addresses: list[str]) -> None:
    """Force :func:`stdapi.security._resolve_hostname` to return *addresses*."""

    async def _fake(_: str) -> list[str]:
        return addresses

    monkeypatch.setattr(security, "_resolve_hostname", _fake)


async def test_validate_host_ssrf_blocks_hostname_resolving_to_unsafe_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named host resolving to a loopback/private address is rejected."""
    _patch_resolution(monkeypatch, ["127.0.0.1"])
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf("example.test")
    assert exc.value.status == 403


async def test_validate_host_ssrf_allows_hostname_resolving_to_public_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named host resolving only to public addresses passes validation."""
    _patch_resolution(monkeypatch, ["93.184.216.34"])
    assert await security.validate_host_ssrf("example.test") == ["93.184.216.34"]


async def test_validate_host_ssrf_fails_closed_on_unresolvable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that resolves to no address is rejected (fail closed)."""
    _patch_resolution(monkeypatch, [])
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf("nonexistent.test")
    assert exc.value.status == 403


async def test_validate_host_ssrf_rejects_unparseable_resolved_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolved address that cannot be parsed is treated as unsafe (fail closed)."""
    _patch_resolution(monkeypatch, ["not-an-ip"])
    with pytest.raises(ApiError) as exc:
        await security.validate_host_ssrf("example.test")
    assert exc.value.status == 403


async def test_ssrf_safe_connector_returns_validating_connector() -> None:
    """The connector factory returns a usable connector backed by the shared resolver."""
    connector = security.ssrf_safe_connector()
    try:
        assert isinstance(connector, TCPConnector)
    finally:
        await connector.close()


class TestSsrfSafeConnectorIntegration:
    """End-to-end SSRF enforcement through the aiohttp connector path."""

    async def test_unsafe_ip_literal_blocked_before_connecting(self) -> None:
        """A request to an unsafe IP literal is rejected without connecting."""
        async with ClientSession(connector=security.ssrf_safe_connector()) as session:
            with pytest.raises(ClientConnectorError) as exc:
                await session.get("http://169.254.169.254/")
        assert isinstance(exc.value.os_error, security.SsrfBlockedError)
        assert security.ssrf_blocked_status(exc.value) == 403

    async def test_hostname_resolving_to_unsafe_ip_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A named host resolving to an unsafe address is rejected at connect time."""
        _patch_resolution(monkeypatch, ["169.254.169.254"])
        async with ClientSession(connector=security.ssrf_safe_connector()) as session:
            with pytest.raises(ClientConnectorError) as exc:
                await session.get("http://internal.test/")
        assert isinstance(exc.value.os_error, security.SsrfBlockedError)

    async def test_redirect_hop_to_unsafe_target_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A redirect from a safe host to an unsafe IP-literal target is blocked at the hop."""
        monkeypatch.setattr(
            security, "_is_unsafe_ip", lambda ip: str(ip) != "127.0.0.1"
        )

        async def redirect(_: web.Request) -> web.Response:
            raise web.HTTPFound(location="http://169.254.169.254/")

        app = web.Application()
        app.router.add_get("/redirect", redirect)
        server = TestServer(app)
        await server.start_server()
        try:
            async with ClientSession(
                connector=security.ssrf_safe_connector()
            ) as session:
                with pytest.raises(ClientConnectorError) as exc:
                    await session.get(server.make_url("/redirect"))
            assert isinstance(exc.value.os_error, security.SsrfBlockedError)
            assert security.ssrf_blocked_status(exc.value) == 403
        finally:
            await server.close()

    async def test_connection_pinned_to_validated_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A named host connects to the resolver-validated address, not system DNS."""
        monkeypatch.setattr(
            security, "_is_unsafe_ip", lambda ip: str(ip) != "127.0.0.1"
        )

        async def ok(_: web.Request) -> web.Response:
            return web.Response(text="pinned")

        app = web.Application()
        app.router.add_get("/ok", ok)
        server = TestServer(app)
        await server.start_server()
        try:
            _patch_resolution(monkeypatch, ["127.0.0.1"])
            async with (
                ClientSession(connector=security.ssrf_safe_connector()) as session,
                session.get(f"http://pinned.test:{server.port}/ok") as resp,
            ):
                assert resp.status == 200
                assert await resp.text() == "pinned"
        finally:
            await server.close()
