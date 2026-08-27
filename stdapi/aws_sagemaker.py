"""Amazon SageMaker AI endpoint HTTP client.

A SageMaker AI real-time endpoint whose container serves the OpenAI Chat
Completions API answers on the ``runtime.sagemaker`` host under
``/endpoints/<endpoint>/openai/v1/...``, with the inference component -- when
the endpoint hosts one -- named in the **path** rather than in a header. The
route exists in no botocore service model, and plain SigV4 is refused: the call
takes the same presigned ``CallWithBearerToken`` bearer token as Amazon Bedrock
Mantle, three constants apart (:mod:`stdapi.aws_http`).

The module also absorbs a scale-to-zero cold start. An endpoint scaled to zero
answers within about a second, and that rejected request is itself what makes
SageMaker AI provision capacity again: the alarm on
``NoCapacityInvocationFailures`` drives the copy count from zero to one, and the
endpoint answers a few minutes later. Nothing here asks AWS to scale up -- the
gateway never writes to a resource the operator owns -- it only declines to give
up, and concurrent callers wait on one shared probe instead of each starting
their own.
"""

from asyncio import Task, create_task, shield, sleep
from collections.abc import Mapping
from contextlib import asynccontextmanager
from re import compile as compile_regex
from time import monotonic
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import quote
from weakref import finalize

from aiohttp import ClientError as AiohttpClientError
from aiohttp import ClientSession, SocketTimeoutError
from pydantic_core import from_json

from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.aws_http import TOKEN_TTL as _TOKEN_TTL
from stdapi.aws_http import iter_sse, new_http_session, presigned_bearer_token
from stdapi.config import AWS_SESSION, SETTINGS
from stdapi.utils import to_json_bytes

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from aiohttp import ClientResponse
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_http import SseEvent

#: The feature a caller is refused, named as the caller knows it.
_FEATURE: Final = "This model"

#: SigV4 signing host used to presign bearer tokens (the control-plane host).
_TOKEN_HOST = "sagemaker.amazonaws.com"  # noqa: S105

#: SigV4 signing name used to presign bearer tokens.
_TOKEN_SERVICE = "sagemaker"  # noqa: S105

#: Bearer token prefix defined by the SageMaker AI API-key format.
_TOKEN_PREFIX = "sagemaker-api-key-"  # noqa: S105

#: Cached bearer tokens: region -> (token, monotonic expiry).
_TOKENS: dict[str, tuple[str, float]] = {}

#: Shared HTTP session toward SageMaker AI endpoints (opened at startup when enabled).
_SESSION: ClientSession | None = None

#: botocore service whose endpoint hosts the OpenAI-compatible invocation routes.
_RUNTIME_SERVICE: Final = "sagemaker-runtime"

#: Runtime host template, per the service's own endpoint rules.
_RUNTIME_HOST: Final = "runtime.sagemaker.{region}.{suffix}"

#: Endpoint resolver, used to build the partition-correct runtime host.
_ENDPOINT_RESOLVER = AWS_SESSION.get_component("endpoint_resolver")

#: Resolved runtime endpoint base URL per region (settings are final after startup).
_ENDPOINT_URLS: dict[str, str] = {}

#: OpenAI-compatible route the supported inference containers serve.
CHAT_COMPLETIONS_PATH: Final = "/openai/v1/chat/completions"

#: Front-door wrapper quoting, verbatim, the body the container itself answered with.
_CONTAINER_BODY_RE = compile_regex(
    r'(?s)^Received (?:client|server) error \(\d+\) from \S+ with message "(.*)"'
)

#: Front-door wording, from its first word, marking an endpoint with no capacity yet.
_NO_CAPACITY_RE = compile_regex(
    r"(?i)^(?:inference component|endpoint)\b[^.]*\bhas no capacity\b"
)

#: A traceback frame, with the deeper-indented source lines belonging to it.
_TRACEBACK_FRAME_RE = compile_regex(
    r'(?m)^([ \t]*)File ["\'][^"\'\n]+["\'], line \d+.*(?:\n\1[ \t]+.*)*'
)

#: The line a traceback opens with, whatever wrote it.
_TRACEBACK_HEADER_RE = compile_regex(
    r"(?im)^[ \t]*Traceback \(most recent call last\):[ \t]*$"
)

#: An absolute filesystem path: rooted, several directories deep, naming a file.
_FILE_PATH_RE = compile_regex(
    r"(?:[A-Za-z]:)?[/\\](?:[\w.+-]+[/\\]){2,}[\w.+-]*\.\w+\b"
)

#: A run of blank lines left behind once a traceback is removed.
_BLANK_LINES_RE = compile_regex(r"\n[ \t]*(?:\n[ \t]*)+")

#: Any word character, the least a sanitized message must keep to be worth showing.
_INTELLIGIBLE_RE = compile_regex(r"\w")

#: What a caller reads when nothing intelligible survives sanitizing.
_CONTAINER_REJECTED: Final = "This model's server rejected the request."

#: Status a request to an endpoint scaled to zero is rejected with.
_NO_CAPACITY_STATUS: Final = 400

#: Seconds between two warm-up probes while an endpoint is provisioning.
_WARMUP_PROBE_INTERVAL: Final = 10.0

#: Times a request is re-sent after a probe reported capacity, before giving up.
_MAX_WARMUP_RETRIES: Final = 3

#: Smallest request proving the endpoint can serve one, used to watch a warm-up.
_WARMUP_PROBE: Final[Mapping[str, Any]] = {
    # Empty: the endpoint and its inference component name the model, in the URL.
    "model": "",
    "messages": [{"role": "user", "content": "."}],
    "max_tokens": 1,
    "stream": False,
}

#: One warm-up probe per cold endpoint: (region, endpoint, component) -> the watcher.
_WARMING: dict[tuple[str, str, str], Task[bool]] = {}


class SageMakerError(ApiError):
    """SageMaker AI upstream error mapped to an API error.

    Attributes:
        no_capacity: Whether the endpoint rejected the request for want of
            capacity, which is the signal that a scale-from-zero has started.
    """

    no_capacity: bool = False

    def __init__(
        self, message: str, *, status: int | None = None, no_capacity: bool = False
    ) -> None:
        """Create a SageMaker AI error.

        Args:
            message: Human-readable error message.
            status: Optional HTTP status code override.
            no_capacity: Whether the endpoint has no capacity yet.
        """
        super().__init__(message, status=status)
        self.no_capacity = no_capacity


def endpoint_url(region: RegionName) -> str:
    """Return the SageMaker AI runtime base URL for *region*.

    Resolved through the AWS SDK so an endpoint in the AWS GovCloud, China or
    European Sovereign Cloud partition gets its own hostname, which is where
    this backend reaches further than Amazon Bedrock Mantle does. Only the
    partition's DNS suffix is taken from the resolver: the host it answers for
    this service is the legacy ``sagemaker-runtime.<region>.<suffix>``, which
    does not serve the OpenAI-compatible routes.

    Args:
        region: AWS region name.

    Returns:
        Base URL without a trailing slash.

    Raises:
        FeatureUnavailableError: When SageMaker AI has no endpoint in *region*.
    """
    if template := SETTINGS.aws_sagemaker_endpoint_url:
        return template.format(region=region).rstrip("/")
    if (cached := _ENDPOINT_URLS.get(region)) is not None:
        return cached
    resolved = _ENDPOINT_RESOLVER.construct_endpoint(_RUNTIME_SERVICE, region)
    if not resolved:
        raise FeatureUnavailableError(
            _FEATURE,
            f"Amazon SageMaker AI has no endpoint in {region}: correct the "
            "region of the matching aws_sagemaker_endpoints entry, or set "
            "aws_sagemaker_endpoint_url to the host that serves it.",
        )
    host = _RUNTIME_HOST.format(region=region, suffix=resolved["dnsSuffix"])
    url = f"https://{host}"
    _ENDPOINT_URLS[region] = url
    return url


def invocation_path(endpoint: str, inference_component: str = "") -> str:
    """Build the request path invoking an endpoint's OpenAI-compatible route.

    The inference component belongs in the path: the endpoint rejects the
    ``X-Amzn-SageMaker-Inference-Component`` header whatever its value.

    Args:
        endpoint: Endpoint name.
        inference_component: Inference component name, when the endpoint hosts
            components rather than a model directly.

    Returns:
        Request path, absolute from the runtime host.
    """
    path = f"/endpoints/{quote(endpoint, safe='')}"
    if inference_component:
        path += f"/inference-components/{quote(inference_component, safe='')}"
    return f"{path}{CHAT_COMPLETIONS_PATH}"


async def bearer_token(region: RegionName) -> str:
    """Return a short-term SageMaker AI bearer token for *region*.

    Args:
        region: AWS region the token is scoped to.

    Returns:
        Bearer token string.

    Raises:
        ApiError: When no AWS credentials are available.
    """
    if (cached := _TOKENS.get(region)) and cached[1] > monotonic():
        return cached[0]
    token = await presigned_bearer_token(
        region,
        host=_TOKEN_HOST,
        service=_TOKEN_SERVICE,
        prefix=_TOKEN_PREFIX,
        feature="SageMaker AI",
    )
    _TOKENS[region] = (token, monotonic() + _TOKEN_TTL)
    return token


@asynccontextmanager
async def sagemaker_http_session() -> AsyncGenerator[ClientSession]:
    """Open the shared SageMaker AI HTTP session for the server's lifetime.

    Only the first opener owns the module-level session; a later one reuses it
    and leaves it untouched on exit, so the server's session is never closed
    from under it. A warm-up probe still sleeping is cancelled rather than
    dropped: an orphan would wake into the next lifespan's session and send one
    request from a lifecycle that no longer owns it.

    Yields:
        The shared :class:`aiohttp.ClientSession`.
    """
    global _SESSION  # noqa: PLW0603
    if _SESSION is not None:
        yield _SESSION
        return
    session = new_http_session()
    _SESSION = session
    try:
        yield session
    finally:
        _SESSION = None
        _TOKENS.clear()
        for probe in tuple(_WARMING.values()):
            probe.cancel()
        _WARMING.clear()
        await session.close()


def _sanitize_container_message(message: str) -> str:
    """Strip from a container's own message what its operator did not publish.

    What the container says about the request is the caller's -- it is the only
    account of why the request was refused. The traceback it may append is not:
    the frames disclose the runtime version and the library layout of a server
    the caller has no business seeing, to anyone able to send a malformed
    request (AGENTS.md, *Never Leak Internals*).

    Matched on the shape of a traceback rather than on any one container's
    wording, so a deployment serving something other than vLLM is covered too.

    Args:
        message: The message the container itself wrote.

    Returns:
        The message without its traceback frames or filesystem paths, or a
        message naming nothing when that leaves nothing worth reading.
    """
    text = _TRACEBACK_FRAME_RE.sub("", message)
    text = _TRACEBACK_HEADER_RE.sub("", text)
    text = _FILE_PATH_RE.sub("...", text)
    text = _BLANK_LINES_RE.sub("\n", text).strip()
    return text if _INTELLIGIBLE_RE.search(text) else _CONTAINER_REJECTED


def _error_details(body: str) -> tuple[str, dict[str, Any], bool]:
    """Extract the message, the error object and who wrote them.

    Two writers answer on this route and only one of them describes the
    caller's own request, but the shape does not tell them apart: measured
    against a real endpoint, the front door answers OpenAI's ``{"error": ...}``
    envelope for its own refusals as well. What it does instead is quote the
    container's whole body inside its own message, and that quotation is the
    only mark of authorship there is -- an unquoted body is the front door's,
    and names the endpoint, its inference component, the account they live in
    and a console URL for the operator's logs. What the container wrote is
    sanitized on the way out, since it may carry a traceback of its own.

    Args:
        body: Raw response body.

    Returns:
        Tuple of (message, error object carrying ``code``/``param`` if any,
        whether the container wrote it rather than the front door).
    """
    try:
        payload: Any = from_json(body)
    except ValueError:
        return body.strip(), {}, False
    if not isinstance(payload, Mapping):
        return body.strip(), {}, False
    details = payload.get("error")
    if isinstance(details, str):
        return details, {}, False
    if not isinstance(details, Mapping):
        details = payload
    message = details.get("message") or details.get("Message") or ""
    if not isinstance(message, str):
        return "", dict(details), False
    if quoted := _CONTAINER_BODY_RE.match(message):
        # The container's own body, quoted whole: read it as the answer. Every
        # message this module forwards comes through here, and none goes
        # further without being sanitized.
        inner_message, inner_details, _ = _error_details(quoted[1])
        return _sanitize_container_message(inner_message), inner_details, True
    return message, dict(details), False


def _map_error(status: int, body: str, region: RegionName) -> ApiError:
    """Map a SageMaker AI HTTP error response to an API error.

    Only what the container wrote is forwarded: it describes the request the
    caller sent. The front door's own body goes to the operator's log and the
    caller reads our envelope instead, since it carries the endpoint name and
    the deployment's account ID (AGENTS.md, *Never Leak Internals*).

    The no-capacity classification is deliberately narrow -- anything it
    matches holds the caller for minutes -- so it takes the front door's own
    wording, from the first word of the message, on the one status it uses. It
    cannot also require the front door's own shape: measured against a real
    endpoint at zero copies, the rejection arrives inside OpenAI's ``error``
    envelope, exactly as a container error does.

    Args:
        status: HTTP status code.
        body: Raw response body.
        region: Region the request was sent to.

    Returns:
        The mapped error, ready to raise.
    """
    message, details, from_container = _error_details(body)
    if status == _NO_CAPACITY_STATUS and _NO_CAPACITY_RE.match(message):
        # Not surfaced while a budget remains: the caller of
        # _request_with_warmup waits on this instead of raising it. Surfaced,
        # it carries 503 rather than the front door's 400: the condition is
        # transient, and no SDK retries a 400 -- the same answer the
        # bedrock-runtime ModelNotReadyException and the exhausted warm-up
        # budget already give.
        return SageMakerError(
            "The model is starting up. Retry in a few minutes.",
            status=503,
            no_capacity=True,
        )
    if status in (401, 403):
        # Server-side credential or permission issue: evict the cached token so
        # a rotated credential self-heals.
        _TOKENS.pop(region, None)
        return FeatureUnavailableError(
            _FEATURE,
            f"{message or f'HTTP {status}'} The server role needs "
            "sagemaker:CallWithBearerToken and sagemaker:InvokeEndpoint on this "
            "endpoint.",
        )
    if not from_container:
        return _front_door_error(status, message or body.strip())
    error = SageMakerError(message or _generic_message(status), status=status)
    if code := details.get("code"):
        error.code = str(code)
    if param := details.get("param"):
        error.param = str(param)
    return error


def _generic_message(status: int) -> str:
    """Return the message a caller reads when the upstream one cannot be shown.

    Args:
        status: HTTP status code.

    Returns:
        A message naming nothing of the backend.
    """
    if status == 429 or status >= 500:
        return "The service is temporarily unavailable. Retry the request."
    return f"The request could not be completed (HTTP {status})."


def _front_door_error(status: int, detail: str) -> SageMakerError:
    """Map an error the SageMaker AI front door wrote, keeping its text private.

    Args:
        status: HTTP status code.
        detail: The upstream text, for the operator's log only.

    Returns:
        The mapped error, ready to raise.
    """
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import log_error_details  # noqa: PLC0415

    log_error_details(
        f"The Amazon SageMaker AI endpoint refused the invocation with HTTP "
        f"{status}: {detail or '(no message)'}",
        level="warning",
    )
    return SageMakerError(_generic_message(status), status=status)


async def _request(
    region: RegionName, path: str, body: bytes, headers: Mapping[str, str] | None = None
) -> ClientResponse:
    """Send one request to a SageMaker AI endpoint and validate its status.

    Args:
        region: Region the endpoint lives in.
        path: Request path, absolute from the runtime host.
        body: Serialized JSON request body.
        headers: Extra HTTP headers.

    Returns:
        The open :class:`aiohttp.ClientResponse` (status already checked).

    Raises:
        ApiError: On upstream errors or connection failures, mapped by
            :func:`_map_error`.
    """
    if _SESSION is None:  # pragma: no cover
        raise FeatureUnavailableError(
            _FEATURE,
            "Amazon SageMaker AI support is not initialized on this server: "
            "aws_sagemaker_endpoints was empty when the lifespan started.",
        )
    try:
        response = await _SESSION.post(
            f"{endpoint_url(region)}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {await bearer_token(region)}",
                "Content-Type": "application/json",
                **(headers or {}),
            },
        )
    except SocketTimeoutError as error:
        # Imported here: stdapi.monitoring transitively imports this module.
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        log_error_details(f"Timed out reading the model response in {region}.")
        msg = "The service is temporarily unavailable. Retry the request."
        raise SageMakerError(msg, status=503) from error
    except (AiohttpClientError, TimeoutError) as error:
        # Imported here: stdapi.monitoring transitively imports this module.
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        log_error_details(f"Unable to reach the model endpoint in {region}.")
        msg = "The service is temporarily unavailable. Retry the request."
        raise SageMakerError(msg, status=503) from error
    if response.status >= 400:
        try:
            error_body = await response.text()
        except AiohttpClientError, TimeoutError:  # pragma: no cover
            error_body = ""
        finally:
            response.release()
        raise _map_error(response.status, error_body, region)
    return response


async def _watch_warm_up(
    region: RegionName, endpoint: str, inference_component: str, deadline: float
) -> bool:
    """Probe a cold endpoint until it can serve, or the budget runs out.

    One probe watches the scale-up for every waiter: the request that already
    failed has triggered it, and polling it N times would only add load to an
    endpoint that has none to give. Never raises, so a waiter that goes away
    leaves nothing unretrieved behind it.

    Args:
        region: Region the endpoint lives in.
        endpoint: Endpoint name.
        inference_component: Inference component name, if any.
        deadline: Monotonic time the warm-up budget expires at.

    Returns:
        ``True`` once the endpoint answers with anything other than "no
        capacity", ``False`` when the budget expired first.
    """
    path = invocation_path(endpoint, inference_component)
    body = to_json_bytes(_WARMUP_PROBE)
    while True:
        if (remaining := deadline - monotonic()) <= 0:
            return False
        await sleep(min(_WARMUP_PROBE_INTERVAL, remaining))
        try:
            response = await _request(region, path, body)
        except SageMakerError as error:
            if error.no_capacity:
                continue
            # Anything else is the real answer: let the waiters see it
            # themselves rather than reporting it from a probe. Cancellation is
            # a BaseException and propagates, ending the probe with it.
            return True
        except Exception:  # noqa: BLE001 - the waiters re-send and surface it
            return True
        response.release()
        return True


async def _wait_for_capacity(
    region: RegionName, endpoint: str, inference_component: str, deadline: float
) -> bool:
    """Wait on the one warm-up watching this endpoint, starting it if needed.

    The watcher is detached from every request: a client that disconnects
    cancels its own wait and never the shared probe, so the first caller
    hanging up does not strand the others.

    Args:
        region: Region the endpoint lives in.
        endpoint: Endpoint name.
        inference_component: Inference component name, if any.
        deadline: Monotonic time the warm-up budget expires at.

    Returns:
        ``True`` when the endpoint became able to answer, ``False`` on timeout.
    """
    key = (region, endpoint, inference_component)
    task = _WARMING.get(key)
    if task is None or task.done():
        task = create_task(
            _watch_warm_up(region, endpoint, inference_component, deadline)
        )
        _WARMING[key] = task
        task.add_done_callback(
            lambda done: _WARMING.pop(key, None) if _WARMING.get(key) is done else None
        )
    return await shield(task)


def _warm_up_timeout_error(
    region: RegionName, endpoint: str, inference_component: str, budget: float
) -> SageMakerError:
    """Build the error a caller gets when the warm-up budget runs out.

    Args:
        region: Region the endpoint lives in.
        endpoint: Endpoint name.
        inference_component: Inference component name, if any.
        budget: Warm-up budget that elapsed, in seconds.

    Returns:
        The error to raise, its real cause already logged for the operator.
    """
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import log_error_details  # noqa: PLC0415

    target = f"'{endpoint}'" + (
        f" (inference component '{inference_component}')" if inference_component else ""
    )
    log_error_details(
        f"The SageMaker AI endpoint {target} in {region} still had no capacity "
        f"after {budget:g}s. An endpoint scaled to zero only provisions capacity "
        "when a CloudWatch alarm on NoCapacityInvocationFailures triggers its "
        "step scaling policy: check that the alarm exists and is not in "
        "INSUFFICIENT_DATA, or raise aws_sagemaker_warmup_timeout.",
        level="warning",
    )
    return SageMakerError(
        "The model is starting up and did not become available in time. "
        "Retry in a few minutes.",
        status=503,
    )


async def _request_with_warmup(
    region: RegionName,
    endpoint: str,
    inference_component: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> ClientResponse:
    """Send a request, absorbing a scale-from-zero cold start.

    The wait happens before the response object exists, so a streaming request
    pays a longer time to first byte and nothing else: no partial stream, and a
    real HTTP status when the budget is exhausted.

    Args:
        region: Region the endpoint lives in.
        endpoint: Endpoint name.
        inference_component: Inference component name, if any.
        payload: JSON request body.
        headers: Extra HTTP headers.

    Returns:
        The open :class:`aiohttp.ClientResponse`.

    Raises:
        SageMakerError: On upstream errors, or when the endpoint did not become
            available within ``aws_sagemaker_warmup_timeout``.
    """
    path = invocation_path(endpoint, inference_component)
    body = to_json_bytes(payload)
    budget = SETTINGS.aws_sagemaker_warmup_timeout
    deadline = monotonic() + budget
    retries = _MAX_WARMUP_RETRIES
    while True:
        try:
            return await _request(region, path, body, headers)
        except SageMakerError as error:
            if not error.no_capacity or budget <= 0:
                raise
            if monotonic() >= deadline or not await _wait_for_capacity(
                region, endpoint, inference_component, deadline
            ):
                raise _warm_up_timeout_error(
                    region, endpoint, inference_component, budget
                ) from error
            # A probe that reported capacity and a request still refused is a
            # race for the one copy that just came up; a few rounds cover it,
            # and an unbounded loop would hold the caller for the whole budget
            # on a rejection this only looks like.
            retries -= 1
            if retries < 0:
                raise


async def invoke(
    region: RegionName,
    endpoint: str,
    inference_component: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Invoke an endpoint's Chat Completions route and return the parsed response.

    Args:
        region: Region the endpoint lives in.
        endpoint: Endpoint name.
        inference_component: Inference component name, if any.
        payload: JSON request body.
        headers: Extra HTTP headers.

    Returns:
        Parsed JSON response body.

    Raises:
        SageMakerError: On upstream errors.
    """
    response = await _request_with_warmup(
        region, endpoint, inference_component, payload, headers
    )
    async with response:
        try:
            return await response.json(  # type: ignore[no-any-return]
                content_type=None, loads=from_json
            )
        except (AiohttpClientError, TimeoutError, ValueError) as error:
            msg = "The model response could not be read."
            raise SageMakerError(msg, status=502) from error


async def invoke_stream(
    region: RegionName,
    endpoint: str,
    inference_component: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[SseEvent]:
    """Invoke an endpoint's Chat Completions route and stream its events.

    Args:
        region: Region the endpoint lives in.
        endpoint: Endpoint name.
        inference_component: Inference component name, if any.
        payload: JSON request body.
        headers: Extra HTTP headers.

    Returns:
        Async generator of ``(event name or None, raw data)`` tuples. Its
        ``async with`` only runs once iteration starts, so a GC-tied fallback
        closes the response for a generator dropped without ever being iterated.

    Raises:
        SageMakerError: On upstream errors before the stream opens.
    """
    response = await _request_with_warmup(
        region, endpoint, inference_component, payload, headers
    )
    generator = iter_sse(
        response, "The model response stream was interrupted.", SageMakerError
    )
    finalize(generator, response.close)
    return generator
