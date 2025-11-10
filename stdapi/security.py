"""Security related utilities."""

from asyncio import Lock, as_completed, create_task
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Literal
from urllib.parse import urlparse

from aiodns import DNSResolver
from aiodns.error import DNSError
from fastapi import HTTPException

from stdapi.config import SETTINGS

_RESOLVER_CACHE: dict[Literal["DNS"], DNSResolver] = {}
_RESOLVER_LOCK = Lock()


async def validate_url_ssrf(url: str | None) -> None:
    """Validate URL to avoid SSRF attacks.

    This function concurrently validates the domain name for "A" and "AAAA" record types using
    asynchronous tasks. It ensures that the hostname has valid DNS records for these types.

    Args:
        url: The URL to validate.
    """
    parsed = urlparse(url).hostname
    hostname = parsed.decode() if isinstance(parsed, bytes) else parsed
    if hostname:
        async with _RESOLVER_LOCK:
            try:
                resolver = _RESOLVER_CACHE["DNS"]
            except KeyError:
                resolver = _RESOLVER_CACHE["DNS"] = DNSResolver()
        for task in as_completed(
            (
                create_task(_validate_hostname(resolver, hostname, "A")),
                create_task(_validate_hostname(resolver, hostname, "AAAA")),
            )
        ):
            await task


def _is_unsafe_ip(ip: int | str | bytes | IPv4Address | IPv6Address | None) -> bool:
    """Validate the IP to avoid SSRF attacks.

    This function checks if an IP address is considered unsafe for making
    external requests. It performs multiple security checks including:
    - Loopback addresses (127.0.0.1, ::1)
    - Unspecified addresses (0.0.0.0, ::)
    - Link-local addresses
    - Reserved addresses
    - Multicast addresses
    - Private network addresses (when enabled in config)

    Args:
        ip: IP address string to check.

    Returns:
        bool: True if the IP is unsafe, False otherwise.
    """
    try:
        address = ip_address(ip)  # type: ignore[arg-type]
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or (SETTINGS.ssrf_protection_block_private_networks and address.is_private)
    )


async def _validate_hostname(
    resolver: DNSResolver, hostname: str, query_type: Literal["A", "AAAA"]
) -> None:
    """Validates a domain by ensuring it does not resolve to an unsafe IP address.

    Performs a domain name system (DNS) query for the given hostname and query type.
    If the resolved IP is deemed unsafe, an HTTPException is raised.
    If the DNS query fails, the process silently ignores the error and returns None.

    Args:
        resolver: DNS resolver instance.
        hostname: The domain name to be validated.
        query_type: The type of DNS query to perform, either "A"
            for IPv4 or "AAAA" for IPv6.

    Raises:
        HTTPException: If the hostname resolves to an unsafe IP address.
    """
    try:
        for rdata in await resolver.query(hostname, query_type):
            if _is_unsafe_ip(rdata.host):
                raise HTTPException(
                    status_code=403, detail=f"Forbidden hostname in URL: {hostname}."
                )
    except DNSError:
        return
