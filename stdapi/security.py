"""Security related utilities."""

from asyncio import Lock
from ipaddress import IPv4Address, IPv6Address, ip_address
from socket import AF_INET, AF_INET6, AF_UNSPEC, AI_NUMERICHOST, SOCK_STREAM
from typing import TYPE_CHECKING, Literal

from aiodns import DNSResolver
from aiodns.error import DNSError
from aiohttp import ClientConnectorError, ClientError, TCPConnector
from aiohttp.abc import AbstractResolver, ResolveResult

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aiohttp.tracing import Trace


class SsrfBlockedError(OSError):
    """Connection target rejected by SSRF validation."""


_RESOLVER_CACHE: dict[Literal["DNS"], DNSResolver] = {}
_RESOLVER_LOCK = Lock()


async def validate_host_ssrf(hostname: str) -> list[str]:
    """Return the safe addresses for *hostname*, rejecting SSRF targets.

    IP-literal hosts are checked directly; named hosts are resolved for both
    "A" and "AAAA" records and every returned address is checked. Fails closed:
    a host that cannot be resolved to any address is rejected.

    Args:
        hostname: The host to validate.

    Returns:
        The validated IP addresses the host is allowed to connect to.

    Raises:
        ApiError: If the host is an unsafe IP or resolves to one (403), or
            cannot be resolved at all (400, an invalid URL rather than a
            policy block).
    """
    if (literal := _parse_ip_literal(hostname)) is not None:
        if _is_unsafe_ip(str(literal)):
            msg = f"Forbidden host in URL: {hostname}."
            raise ApiError(msg, status=403)
        return [str(literal)]

    if not (addresses := await _resolve_hostname(hostname)):
        msg = f"Cannot resolve host in URL: {hostname}."
        raise ApiError(msg, status=400)
    for address in addresses:
        if _is_unsafe_ip(address):
            msg = f"Forbidden host in URL: {hostname}."
            raise ApiError(msg, status=403)
    return addresses


def _parse_ip_literal(hostname: str) -> IPv4Address | IPv6Address | None:
    """Return the address when *hostname* is an IP literal, else ``None``.

    Args:
        hostname: Host component of a URL, possibly a bracketed IPv6 literal.

    Returns:
        The parsed address, or ``None`` when *hostname* is a name.
    """
    try:
        return ip_address(hostname[1:-1] if hostname.startswith("[") else hostname)
    except ValueError:
        return None


def _is_unsafe_ip(ip: int | str | bytes | IPv4Address | IPv6Address | None) -> bool:
    """Validate the IP to avoid SSRF attacks.

    Rejects loopback, unspecified, link-local, reserved and multicast addresses,
    plus — when enabled in config — every non-globally-reachable address, which
    covers private ranges and the other special-purpose blocks such as RFC 6598
    shared address space (100.64.0.0/10).

    Args:
        ip: IP address string to check.

    Returns:
        bool: True if the IP is unsafe, False otherwise.
    """
    try:
        address = ip_address(ip)  # type: ignore[arg-type]
    except ValueError:
        return True
    return bool(
        address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or (SETTINGS.ssrf_protection_block_private_networks and not address.is_global)
    )


async def _resolve_hostname(hostname: str) -> list[str]:
    """Resolve *hostname* to its IPv4 and IPv6 addresses.

    Args:
        hostname: The domain name to resolve.

    Returns:
        All resolved addresses across both families; empty when the host does
        not resolve.
    """
    async with _RESOLVER_LOCK:
        try:
            resolver = _RESOLVER_CACHE["DNS"]
        except KeyError:
            resolver = _RESOLVER_CACHE["DNS"] = DNSResolver()
    try:
        result = await resolver.getaddrinfo(
            hostname, family=AF_UNSPEC, type=SOCK_STREAM
        )
    except DNSError:
        return []
    return [
        address.decode() if isinstance(address := node.addr[0], bytes) else address
        for node in result.nodes
    ]


class _SsrfSafeResolver(AbstractResolver):
    """aiohttp resolver that validates every connection target against SSRF.

    Because validation happens where aiohttp establishes the TCP connection, it
    covers the original request and every redirect hop, and pins the connection
    to a freshly validated address — closing the DNS-rebinding window between
    validation and connect.
    """

    __slots__ = ()

    async def resolve(
        self, host: str, port: int = 0, family: int = AF_INET
    ) -> list[ResolveResult]:
        """Resolve *host*, returning only SSRF-safe addresses.

        Args:
            host: Hostname aiohttp wants to connect to.
            port: Target port.
            family: Requested address family, or ``AF_UNSPEC`` for any.

        Returns:
            Resolved results limited to validated addresses.

        Raises:
            SsrfBlockedError: When *host* is an unsafe SSRF target.
            OSError: When *host* cannot be resolved (a plain DNS failure,
                surfaced as a connection error rather than a policy block).
        """
        try:
            addresses = await validate_host_ssrf(host)
        except ApiError as exc:
            if exc.status == 403:
                raise SsrfBlockedError(exc.args[0]) from exc
            raise OSError(exc.args[0]) from exc
        results: list[ResolveResult] = []
        for ip in addresses:
            ip_family = AF_INET6 if ip_address(ip).version == 6 else AF_INET
            if family in (AF_UNSPEC, ip_family):
                results.append(
                    ResolveResult(
                        hostname=host,
                        host=ip,
                        port=port,
                        family=ip_family,
                        proto=0,
                        flags=AI_NUMERICHOST,
                    )
                )
        return results

    async def close(self) -> None:
        """Release resolver resources (nothing to release)."""


#: Shared, stateless SSRF-validating resolver reused across connectors.
_SHARED_RESOLVER: AbstractResolver = _SsrfSafeResolver()


class _SsrfSafeConnector(TCPConnector):
    """``TCPConnector`` that also validates IP-literal connection targets.

    aiohttp resolves IP-literal hosts internally without consulting the
    configured resolver, so ``_SsrfSafeResolver`` alone never sees them —
    including literal redirect targets. This override validates them before
    delegating to aiohttp's resolution.
    """

    async def _resolve_host(
        self, host: str, port: int, traces: Sequence[Trace] | None = None
    ) -> list[ResolveResult]:
        """Validate IP-literal hosts, then delegate to aiohttp's resolution.

        Args:
            host: Hostname or IP literal aiohttp wants to connect to.
            port: Target port.
            traces: aiohttp trace contexts.

        Returns:
            Resolved results limited to validated addresses.

        Raises:
            SsrfBlockedError: When *host* is an unsafe IP literal.
        """
        if _parse_ip_literal(host) is not None:
            try:
                await validate_host_ssrf(host)
            except ApiError as exc:
                raise SsrfBlockedError(exc.args[0]) from exc
        return await super()._resolve_host(host, port, traces)


def ssrf_safe_connector() -> TCPConnector:
    """Return a ``TCPConnector`` that validates every connection against SSRF.

    A fresh connector is returned per call (owned and closed by its
    ``ClientSession``), while the stateless resolver that performs the actual
    validation is shared. Safe to use with redirects enabled: each hop —
    named host or IP literal — is validated before the connection is made.

    Returns:
        An SSRF-validating connector.
    """
    return _SsrfSafeConnector(resolver=_SHARED_RESOLVER)


def ssrf_blocked_status(error: ClientError) -> int:
    """Return the API status to report for an aiohttp client error.

    Args:
        error: Error raised while performing an HTTP request.

    Returns:
        403 when the error wraps an SSRF rejection, else the default 400.
    """
    if isinstance(error, ClientConnectorError) and isinstance(
        error.os_error, SsrfBlockedError
    ):
        return 403
    return 400
