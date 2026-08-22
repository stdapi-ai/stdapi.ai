"""Minimal HTTP helpers for the generator.

Standard library only: the generator runs from the ``docs`` dependency group and
must not drag a request stack into the documentation build environment.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

#: Identifies the generator to the public APIs it reads.
USER_AGENT: str = "stdapi.ai-model-catalog-generator/1.0 (+https://stdapi.ai)"

#: Schemes the generator is allowed to fetch; plain HTTP only for a local gateway.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Hosts a plain-HTTP URL may point at.
_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

#: Delay before each retry, in seconds; also the cadence of a 503 startup wait.
_RETRY_DELAYS: tuple[float, ...] = (2.0, 5.0, 15.0, 30.0)

#: Concurrent requests issued against a single upstream host.
_MAX_WORKERS: int = 8


class FetchError(RuntimeError):
    """A URL could not be read after every retry."""


class _CheckedRedirects(HTTPRedirectHandler):
    """Applies the scheme rules to every hop, not only the first.

    ``urlopen`` follows redirects itself, and its default handler permits
    ``http`` and ``ftp`` as well as ``https`` — so a one-shot check on the URL
    the caller passed lets an HTTPS source redirect to somewhere the rules
    would have refused.
    """

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        """Build the request for one redirect hop.

        Args:
            req: The request being redirected.
            fp: The response body.
            code: HTTP status code.
            msg: HTTP status message.
            headers: Response headers.
            newurl: Where the response points.

        Returns:
            The request to follow, or ``None`` to stop.

        Raises:
            ValueError: The hop targets a scheme or host the rules refuse.
        """
        _validated(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


#: Opener that re-checks every redirect hop against the scheme rules.
_OPENER = build_opener(_CheckedRedirects)


def _validated(url: str) -> str:
    """Return *url* if the generator is allowed to fetch it.

    Args:
        url: Absolute URL to check.

    Returns:
        The URL unchanged.

    Raises:
        ValueError: The scheme is not allowed, or plain HTTP targets a remote host.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        msg = f"refusing to fetch {parsed.scheme!r} URL: {url}"
        raise ValueError(msg)
    if parsed.scheme == "http" and (parsed.hostname or "") not in _LOCAL_HOSTS:
        msg = f"refusing plain HTTP to a remote host: {url}"
        raise ValueError(msg)
    return url


def get_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
    retry_on: Sequence[int] = (429, 500, 502, 503, 504),
) -> bytes:
    """Fetch a URL, retrying the status codes that mean "try again".

    Args:
        url: Absolute URL to fetch.
        timeout: Per-attempt socket timeout, in seconds.
        headers: Extra request headers.
        retry_on: HTTP status codes worth retrying.

    Returns:
        The response body.

    Raises:
        FetchError: Every attempt failed.
    """
    request = Request(  # noqa: S310 -- scheme checked by _validated
        _validated(url), headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    last: Exception | None = None
    for delay in (0.0, *_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                return bytes(response.read())
        except HTTPError as error:
            last = error
            if error.code not in retry_on:
                break
        except (URLError, TimeoutError, OSError) as error:
            last = error
    msg = f"could not fetch {url}: {last}"
    raise FetchError(msg) from last


def get_json(
    url: str, *, timeout: float = 120.0, headers: dict[str, str] | None = None
) -> Any:  # noqa: ANN401
    """Fetch a URL and decode its JSON body.

    Args:
        url: Absolute URL to fetch.
        timeout: Per-attempt socket timeout, in seconds.
        headers: Extra request headers.

    Returns:
        The decoded JSON document.
    """
    return json.loads(get_bytes(url, timeout=timeout, headers=headers))


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
) -> Any:  # noqa: ANN401
    """Send a JSON body and decode the JSON response.

    Args:
        url: Absolute URL to post to.
        payload: Body to send.
        timeout: Socket timeout, in seconds.
        headers: Extra request headers.

    Returns:
        The decoded JSON document.

    Raises:
        FetchError: The request failed.
    """
    request = Request(  # noqa: S310 -- scheme checked by _validated
        _validated(url),
        data=json.dumps(payload).encode(),
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        detail = (
            error.read().decode("utf-8", "replace")[:500]
            if isinstance(error, HTTPError)
            else ""
        )
        msg = f"could not post to {url}: {error} {detail}".strip()
        raise FetchError(msg) from error


def map_concurrent[T, R](function: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """Apply *function* to every item concurrently, preserving input order.

    Args:
        function: Callable applied to each item.
        items: Items to process.

    Returns:
        The results, in the order of *items*.
    """
    with ThreadPoolExecutor(_MAX_WORKERS) as pool:
        return list(pool.map(function, items))
