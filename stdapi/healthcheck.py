"""Container health probe, run as ``python -S -m stdapi.healthcheck``.

``-S`` skips ``site`` since nothing outside the standard library is imported,
and the request goes straight to a socket rather than through
:mod:`urllib.request`, whose import chain is 120 modules for one unencrypted
loopback request.
"""

from os import getenv
from socket import create_connection


def resolve_host() -> str:
    """Return a ``Host`` header the server's ``TRUSTED_HOSTS`` check accepts."""
    if not (setting := getenv("TRUSTED_HOSTS", "").strip()):
        return "localhost"

    # Unset by default, so json stays off the common path.
    import json  # noqa: PLC0415

    try:
        parsed = json.loads(setting)
    except ValueError:
        parsed = setting
    match parsed:
        case [first, *_] | str(first):
            host = str(first).strip()
        case _:
            return "localhost"
    if host.startswith("*"):
        return "localhost" if host == "*" else "healthcheck" + host[1:]
    return host or "localhost"


def main() -> int:
    """Probe ``/health`` and return the exit status to leave the process with."""
    with create_connection(
        ("127.0.0.1", int(getenv("GRANIAN_PORT", "8000"))), 4
    ) as connection:
        connection.sendall(
            f"GET /health HTTP/1.1\r\n"
            f"Host: {resolve_host()}\r\n"
            f"Connection: close\r\n\r\n".encode()
        )
        with connection.makefile("rb") as response:
            return 0 if response.readline().split()[1:2] == [b"200"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
