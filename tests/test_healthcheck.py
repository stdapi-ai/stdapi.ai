"""Container health probe in :mod:`stdapi.healthcheck`.

The probe is what the container image declares as its ``HEALTHCHECK``, so a
regression here makes every deployment report unhealthy and roll back. The
exchange runs against a stub server on a real loopback socket rather than a
mocked one: the module writes the request bytes itself, so the framing is part
of what has to be checked.

The ``TRUSTED_HOSTS`` mapping asserts the exact host documented for each form of
the setting, since operators size their allow-list against it.

Ref: stdapi/healthcheck.py
Ref: docs/operations_configuration.md#trusted-hosts
"""

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from stdapi.healthcheck import main, resolve_host

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.local

#: Seconds any socket operation in these tests may block before failing the test.
_TIMEOUT = 10.0

#: Bytes of the probe's request head the stub server reads.
_REQUEST_BYTES = 512


@contextmanager
def _stub_server(status_line: bytes) -> Iterator[tuple[int, list[bytes]]]:
    """Serve one request on a loopback port, answering with *status_line*.

    Args:
        status_line: The HTTP status line to answer with, without its ``CRLF``.

    Yields:
        The port the stub listens on, and the list its request head lands in.
    """
    requests: list[bytes] = []
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(_TIMEOUT)

    def serve() -> None:
        """Accept one connection, record its request head, and answer it."""
        try:
            connection, _ = listener.accept()
        except OSError:  # pragma: no cover - the probe failed to connect
            return
        with connection:
            connection.settimeout(_TIMEOUT)
            requests.append(connection.recv(_REQUEST_BYTES))
            connection.sendall(status_line + b"\r\nContent-Length: 0\r\n\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1], requests
    finally:
        thread.join(_TIMEOUT)
        listener.close()


def test_resolve_host_defaults_to_localhost_when_unset() -> None:
    """An unset TRUSTED_HOSTS leaves the server validating nothing to match."""
    assert resolve_host() == "localhost"


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ('["api.example.com", "www.example.com"]', "api.example.com"),
        ('["*.example.com"]', "healthcheck.example.com"),
        ('["*"]', "localhost"),
        ("[]", "localhost"),
        ('""', "localhost"),
        ("api.example.com", "api.example.com"),
        ('{"host": "api.example.com"}', "localhost"),
        ("[123]", "123"),
    ],
    ids=[
        "first-of-list",
        "wildcard-subdomain",
        "any-host",
        "empty-list",
        "empty-string",
        "bare-string",
        "mapping",
        "number",
    ],
)
def test_resolve_host_reads_trusted_hosts(
    monkeypatch: pytest.MonkeyPatch, setting: str, expected: str
) -> None:
    """Every documented TRUSTED_HOSTS form yields a host the server accepts.

    A wildcard entry is a pattern rather than a name, and a setting the server
    itself would reject must not leave the probe without a host to announce.
    """
    monkeypatch.setenv("TRUSTED_HOSTS", setting)
    assert resolve_host() == expected


def test_probe_succeeds_and_announces_a_trusted_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 exits 0, with the Host header the allow-list's first entry names."""
    monkeypatch.setenv("TRUSTED_HOSTS", '["api.example.com"]')
    with _stub_server(b"HTTP/1.1 200 OK") as (port, requests):
        monkeypatch.setenv("GRANIAN_PORT", str(port))
        assert main() == 0
    head = requests[0].decode()
    assert head.startswith("GET /health HTTP/1.1\r\n")
    assert "\r\nHost: api.example.com\r\n" in head


def test_probe_fails_on_a_rejected_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 exits non-zero, so Host validation cannot silently pass the probe."""
    with _stub_server(b"HTTP/1.1 400 Bad Request") as (port, _):
        monkeypatch.setenv("GRANIAN_PORT", str(port))
        assert main() == 1


def test_probe_fails_when_nothing_listens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed port raises rather than reporting the container healthy."""
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
    monkeypatch.setenv("GRANIAN_PORT", str(port))
    with pytest.raises(OSError, match=r"[Cc]onnection refused"):
        main()
