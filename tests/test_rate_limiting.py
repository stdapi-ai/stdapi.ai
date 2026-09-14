"""Tests for the per-tenant request and token rate limits.

A tenant key may carry a requests-per-minute and a tokens-per-minute limit,
declared on its record or defaulted for the deployment. Admission happens in
the authentication dependency, backed by an atomic counter in the shared
table, so several instances can never admit more than the limit between them;
tokens are estimated at admission and reconciled from the billed usage after
the response. The refusal is the vendors' own ``429 rate_limit_error`` with a
``retry-after``, and every limited response carries the vendors' own rate-limit
headers. A table that cannot be written fails closed once the slots already
granted are used up, as the tenant-key validation does on the same table.

The offline lane runs against the local DynamoDB stand-in; the sandbox lane
proves the atomic grant against the real service.

Ref: stdapi/tenant_rate_limits.py
     https://developers.openai.com/api/docs/guides/rate-limits
     https://platform.claude.com/docs/en/api/rate-limits
"""

from __future__ import annotations

import json
import re
from asyncio import CancelledError, Event, create_task, gather, sleep
from datetime import UTC, datetime
from secrets import token_bytes
from time import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from aiobotocore.session import get_session
from fastapi import Request, WebSocket
from pydantic import ValidationError

import stdapi.auth
from stdapi import tenant_keys, tenant_rate_limits
from stdapi.api_errors import ApiError, FeatureUnavailableError, UnsupportedModelError
from stdapi.auth import (
    AuthenticationHandler,
    authenticate,
    verify_websocket_credentials,
)
from stdapi.aws_dynamodb import (
    EXPIRES_AT_ATTRIBUTE,
    PARTITION_KEY,
    SORT_KEY,
    TableUnavailableError,
    add_to_item,
    get_item,
    item_key,
    put_item,
)
from stdapi.config import SETTINGS, _Settings
from stdapi.monitoring import (
    REQUEST,
    TENANT,
    Tenant,
    flush_usage_log_event,
    log_request_event,
    log_request_stream_event,
)
from stdapi.server import INTERNAL_REQUEST_ID_HEADER, MCP_USER_AGENT
from stdapi.tenant_keys import KEY_PREFIX, reconcile_tenant_keys, resume_tenant
from stdapi.tenant_rate_limits import (
    REQUESTS_ATTRIBUTE,
    STATE_KEY,
    TOKENS_ATTRIBUTE,
    WINDOW_SECONDS,
    admit_tenant_request,
    rate_limit_headers,
    settle_tenant_reservation,
    tenant_retry_after,
    verify_counter_access,
)
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from types_aiobotocore_ssm.client import SSMClient

    from stdapi.aws_dynamodb import Item, ItemValue

pytestmark = pytest.mark.local

#: Parameter Store prefix the offline lane delivers keys under.
_PREFIX = "/stdapi-test/rate-limits"

#: RFC 3339 timestamp, as the Anthropic reset headers carry it.
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

#: Go duration of at most one minute, as the OpenAI reset headers carry it.
_GO_DURATION = re.compile(r"^(?:1m0s|(?:[0-9]|[1-5][0-9])s)$")


class _Clock:
    """A clock the limiter reads in place of the wall clock, moved by hand.

    Pinned one second into the window after the real one, so no test straddles
    a minute boundary it did not ask for, and the resets it publishes lie in
    the future of the real clock.
    """

    __slots__ = ("now",)

    def __init__(self) -> None:
        self.now = float(int(time()) // WINDOW_SECONDS * WINDOW_SECONDS) + (
            WINDOW_SECONDS + 1
        )

    def __call__(self) -> float:
        """Return the pinned time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the pinned time forward."""
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Pin the limiter's wall and monotonic clocks to one movable instant.

    Ref: stdapi/tenant_rate_limits.py:_window
    """
    pinned = _Clock()
    monkeypatch.setattr(tenant_rate_limits, "time", pinned)
    monkeypatch.setattr(tenant_rate_limits, "monotonic", pinned)
    return pinned


def _scope(path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Build an HTTP ASGI scope whose matched route reports *path*.

    Args:
        path: The route path template.
        headers: Request headers to carry.

    Returns:
        The scope, complete enough for the request log to bind it.
    """
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "scheme": "http",
        "server": ("gateway", 80),
        "client": ("127.0.0.1", 1234),
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "route": SimpleNamespace(path_format=path, tags=[]),
    }


def _request_for(path: str = "/v1/chat/completions") -> Request:
    """Build a request whose matched route reports *path*.

    Args:
        path: The route path template.

    Returns:
        The request.
    """
    return Request(_scope(path))


def _websocket_scope(key: str) -> dict[str, Any]:
    """Build the ASGI scope of a Realtime handshake presenting *key*.

    Args:
        key: The tenant key, sent as the SDK sends it.

    Returns:
        The scope, complete enough for the request log to bind it.
    """
    return {
        "type": "websocket",
        "path": "/v1/realtime",
        "raw_path": b"/v1/realtime",
        "query_string": b"",
        "scheme": "ws",
        "server": ("gateway", 80),
        "client": ("127.0.0.1", 1234),
        "headers": [(b"authorization", f"Bearer {key}".encode())],
        "route": SimpleNamespace(path_format="/v1/realtime", tags=[]),
    }


async def _handshake(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serve one Realtime handshake in-process, from the upgrade to the close frame.

    Every model is refused after the credential is admitted, which is the
    cheapest way a handshake ends without billing a turn.

    Args:
        key: The tenant key the client presents.
        monkeypatch: Replaces the model validation for the call.

    Returns:
        The connection's scope, and every message the server sent on it.
    """
    from stdapi import realtime  # noqa: PLC0415

    async def _no_such_model(model_id: str, **_: Any) -> Any:  # noqa: ANN401
        raise UnsupportedModelError(model_id)

    monkeypatch.setattr(realtime, "validate_model", _no_such_model)
    monkeypatch.setattr(realtime, "_SHUTTING_DOWN", False)
    scope = _websocket_scope(key)
    inbox = [{"type": "websocket.connect"}]
    sent: list[dict[str, Any]] = []

    async def _receive() -> dict[str, Any]:
        return inbox.pop() if inbox else {"type": "websocket.disconnect", "code": 1000}

    async def _send(message: Any) -> None:  # noqa: ANN401
        sent.append(dict(message))

    await realtime.serve_realtime_session(WebSocket(scope, _receive, _send), "nope")
    return scope, sent


def _tenant_item(key_id: str, **attributes: ItemValue) -> Item:
    """Build the tenant record the Terraform module would write.

    Args:
        key_id: The tenant's key ID.
        **attributes: Extra record attributes, e.g. the limits.

    Returns:
        The item.
    """
    return {
        PARTITION_KEY: "TENANT",
        SORT_KEY: f"tenant#{key_id}",
        "name": f"tenant-{key_id}",
        **attributes,
    }


@pytest.fixture
async def limited_backend(
    dynamodb_table: str,  # noqa: ARG001 - binds the table stand-in
    clock: _Clock,  # noqa: ARG001 - pins the window every counter read keys on
    moto_dynamodb_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[SSMClient]:
    """Enable tenant keys against the local stand-ins, with no deployment key.

    Yields:
        The SSM client bound to the stand-in, for reading delivered keys back.

    Ref: tests/conftest.py:dynamodb_table
    """
    from stdapi.aws import _CLIENTS  # noqa: PLC0415

    region = SETTINGS.aws_bedrock_regions[0]
    session = get_session()
    async with session.create_client(
        "ssm",
        region_name=region,
        endpoint_url=moto_dynamodb_endpoint,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # noqa: S106 - a local stand-in, not a secret
    ) as ssm_client:
        monkeypatch.setitem(_CLIENTS, "ssm", {region: ssm_client})
        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        monkeypatch.setattr(SETTINGS, "tenant_key_cache_seconds", 60.0)
        monkeypatch.setattr(SETTINGS, "tenant_key_ssm_parameter_prefix", _PREFIX)
        monkeypatch.setattr(SETTINGS, "tenant_rate_limit_requests_per_minute", None)
        monkeypatch.setattr(SETTINGS, "tenant_rate_limit_tokens_per_minute", None)
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant_keys._NEGATIVE.clear()  # noqa: SLF001
        tenant_keys._REPORTED.clear()  # noqa: SLF001
        tenant_rate_limits._LEDGERS.clear()  # noqa: SLF001
        token = TENANT.set(None)
        try:
            yield ssm_client
        finally:
            TENANT.reset(token)
            tenant_keys._CACHE.clear()  # noqa: SLF001
            tenant_keys._NEGATIVE.clear()  # noqa: SLF001
            tenant_keys._REPORTED.clear()  # noqa: SLF001
            tenant_rate_limits._LEDGERS.clear()  # noqa: SLF001


async def _declare_and_mint(
    ssm_client: SSMClient, key_id: str, **attributes: ItemValue
) -> str:
    """Declare one tenant, run the reconciliation, and read back its key.

    Args:
        ssm_client: The Parameter Store stand-in client.
        key_id: The tenant's key ID.
        **attributes: Extra tenant-record attributes.

    Returns:
        The full API key, as the operator would deliver it to the tenant.
    """
    await put_item(_tenant_item(key_id, **attributes))
    await reconcile_tenant_keys()
    value = (
        await ssm_client.get_parameter(Name=f"{_PREFIX}/{key_id}", WithDecryption=True)
    )["Parameter"]["Value"]
    assert value.startswith(f"{KEY_PREFIX}{key_id}-")
    return value


async def _admissions(key: str, count: int, *, concurrent: bool) -> list[int]:
    """Present *key* *count* times and collect the status of every refusal.

    Args:
        key: The tenant key.
        count: How many requests to make.
        concurrent: Run them all at once rather than one after the other.

    Returns:
        One entry per request: 0 when admitted, else the refusal's status.
    """

    async def one() -> int:
        try:
            await authenticate(credentials=None, x_api_key=key, request=_request_for())
        except ApiError as error:
            return error.status
        return 0

    if concurrent:
        return list(await gather(*(one() for _ in range(count))))
    return [await one() for _ in range(count)]


async def _settle(key_id: str) -> None:
    """Wait for the pending token flush and counter write of *key_id*'s ledger."""
    ledger = tenant_rate_limits._LEDGERS.get(key_id)  # noqa: SLF001
    if ledger is None:
        return
    if ledger.flush_task is not None:
        await ledger.flush_task
    if ledger.sync_task is not None:
        await ledger.sync_task


def _current_window() -> int:
    """Start of the window the limiter's (pinned) clock is in."""
    now = tenant_rate_limits.time()  # type: ignore[attr-defined]  # the pinned import
    return tenant_rate_limits._window(now)  # noqa: SLF001


async def _counter(key_id: str, window: int | None = None) -> Item | None:
    """Read the counter item of *key_id* for *window*, else the current one."""
    window = _current_window() if window is None else window
    return await get_item(item_key("LIMIT", key_id), str(window))


async def _bill(key: str, tokens: int, **usage: Any) -> Request:  # noqa: ANN401
    """Admit one request of *key* and record what the model billed it.

    Args:
        key: The tenant key.
        tokens: Input tokens billed, the whole total when no other is given.
        **usage: Further ``record_bedrock_usage`` quantities.

    Returns:
        The request, its reservation on the scope.
    """
    request = _request_for()
    with log_request_event(request):
        await authenticate(credentials=None, x_api_key=key, request=request)
        record_bedrock_usage("amazon.nova-micro-v1:0", input_tokens=tokens, **usage)
    return request


class TestSettings:
    """What the settings model accepts for the deployment-wide defaults.

    Ref: stdapi/config.py:_validate_tenant_keys
    """

    @pytest.mark.parametrize(
        "name",
        [
            "tenant_rate_limit_requests_per_minute",
            "tenant_rate_limit_tokens_per_minute",
        ],
    )
    def test_a_default_below_one_is_refused(self, name: str) -> None:
        """A limit of zero would refuse every request, which is a mistake to name.

        Ref: stdapi/config.py:_Settings.tenant_rate_limit_requests_per_minute
             stdapi/config.py:_Settings.tenant_rate_limit_tokens_per_minute
        """
        overrides: dict[str, Any] = {name: 0}
        with pytest.raises(ValidationError, match=name):
            _Settings(tenant_api_keys=True, aws_dynamodb_table="shared", **overrides)

    @pytest.mark.parametrize(
        "name",
        [
            "tenant_rate_limit_requests_per_minute",
            "tenant_rate_limit_tokens_per_minute",
        ],
    )
    def test_a_default_without_tenant_keys_fails_startup(self, name: str) -> None:
        """Only tenant keys are limited, so a default without them limits nothing.

        Ref: stdapi/config.py:_Settings._validate_tenant_keys
        """
        overrides: dict[str, Any] = {name: 60}
        with pytest.raises(ValidationError, match=r"requires? tenant_api_keys"):
            _Settings(**overrides)

    def test_the_defaults_are_accepted_with_tenant_keys(self) -> None:
        """Both defaults are whole numbers of at least one.

        Ref: stdapi/config.py:_Settings._validate_tenant_keys
             docs/operations_configuration_authentication.md#tenant-rate-limit-requests-per-minute
        """
        settings = _Settings(
            tenant_api_keys=True,
            aws_dynamodb_table="shared",
            tenant_rate_limit_requests_per_minute=60,
            tenant_rate_limit_tokens_per_minute=100000,
        )

        assert settings.tenant_rate_limit_requests_per_minute == 60
        assert settings.tenant_rate_limit_tokens_per_minute == 100000

    @pytest.mark.parametrize(
        ("attribute", "value"),
        [
            ("requests_per_minute", 0),
            ("requests_per_minute", "ten"),
            ("requests_per_minute", True),
            ("tokens_per_minute", 1.5),
            ("tokens_per_minute", -1),
        ],
    )
    async def test_a_malformed_record_limit_is_refused(
        self,
        limited_backend: SSMClient,
        request_log: dict[str, Any],
        attribute: str,
        value: ItemValue,
    ) -> None:
        """A limit the server cannot trust refuses the key and names the attribute.

        Ref: stdapi/tenant_keys.py:_build_entry
        """
        key = await _declare_and_mint(
            limited_backend, key_id="m" + "0" * 15, **{attribute: value}
        )

        with pytest.raises(FeatureUnavailableError) as raised:
            await authenticate(credentials=None, x_api_key=key)

        assert raised.value.status == 503
        assert attribute in str(request_log["error_detail"])
        assert attribute not in str(raised.value)


class TestAdmission:
    """Exactly the limit is admitted, whatever the instance count or concurrency.

    Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
    """

    async def test_exactly_the_limit_is_admitted_across_instances(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two instances sharing the table admit the limit between them, never more.

        A second process is a second ledger dict over the same table: what the
        instances share is the atomic counter, and nothing else.

        Ref: stdapi/tenant_rate_limits.py:_sync
        """
        key = await _declare_and_mint(
            limited_backend, key_id="a" + "0" * 15, requests_per_minute=7
        )

        first = await _admissions(key, 5, concurrent=False)
        await _settle("a" + "0" * 15)
        monkeypatch.setattr(tenant_rate_limits, "_LEDGERS", {})
        second = await _admissions(key, 9, concurrent=False)

        assert first.count(0) + second.count(0) == 7
        assert all(status in {0, 429} for status in first + second)
        assert second[-1] == 429

    async def test_a_concurrent_burst_admits_exactly_the_limit(
        self, limited_backend: SSMClient
    ) -> None:
        """Concurrent requests on one instance cannot slip past the limit together.

        The ledger's check-and-decrement has no await inside it, and the
        waiters on an empty ledger share one counter write.

        Ref: stdapi/tenant_rate_limits.py:_sync_now
             stdapi/tenant_rate_limits.py:admit_tenant_request
        """
        key = await _declare_and_mint(
            limited_backend, key_id="b" + "0" * 15, requests_per_minute=10
        )

        statuses = await _admissions(key, 30, concurrent=True)

        assert statuses.count(0) == 10
        assert statuses.count(429) == 20

    async def test_the_refusal_is_the_vendors_rate_limit_error(
        self, limited_backend: SSMClient
    ) -> None:
        """A 429 naming the key and the limit, with a retry delay inside the minute.

        The message names the key ID (public by design) and the limit, never
        the secret, the table or the counter.

        Ref: https://developers.openai.com/api/docs/guides/rate-limits
        """
        key_id = "c" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=1)
        request = _request_for()

        await authenticate(credentials=None, x_api_key=key, request=request)
        refused = _request_for()
        with pytest.raises(ApiError) as raised:
            await authenticate(credentials=None, x_api_key=key, request=refused)

        assert raised.value.status == 429
        assert raised.value.code == "rate_limit_exceeded"
        message = str(raised.value)
        assert key_id in message
        assert "1 request" in message
        assert key.rsplit("-", 1)[-1] not in message
        assert "dynamodb" not in message.lower()
        assert 1 <= (tenant_retry_after(refused) or 0) <= WINDOW_SECONDS

    async def test_the_record_limit_wins_over_the_deployment_default(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Record, then general configuration: the standing precedence chain.

        Ref: stdapi/tenant_rate_limits.py:resolve_limits
        """
        monkeypatch.setattr(SETTINGS, "tenant_rate_limit_requests_per_minute", 2)
        defaulted = await _declare_and_mint(limited_backend, key_id="d" + "0" * 15)
        declared = await _declare_and_mint(
            limited_backend, key_id="e" + "0" * 15, requests_per_minute=4
        )

        assert (await _admissions(defaulted, 5, concurrent=False)).count(0) == 2
        assert (await _admissions(declared, 5, concurrent=False)).count(0) == 4

    async def test_a_tenant_without_limits_writes_nothing(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No limit means no counter, no write and no header state.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
             stdapi/tenant_rate_limits.py:rate_limit_headers
        """
        writes: list[str] = []

        async def _spy(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401
            writes.append(str(args))
            return await add_to_item(*args, **kwargs)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _spy)
        key = await _declare_and_mint(limited_backend, key_id="f" + "0" * 15)
        request = _request_for()

        for _ in range(5):
            await authenticate(credentials=None, x_api_key=key, request=request)

        assert writes == []
        assert STATE_KEY not in request.scope.get("state", {})
        assert rate_limit_headers(request) == {}

    async def test_a_refused_flood_does_not_write_per_request(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the window is exhausted, refusals cost no table write at all.

        Ref: stdapi/tenant_rate_limits.py:_ensure
        """
        writes: list[str] = []

        async def _spy(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401
            writes.append(str(args))
            return await add_to_item(*args, **kwargs)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _spy)
        key = await _declare_and_mint(
            limited_backend, key_id="g" + "0" * 15, requests_per_minute=3
        )

        await _admissions(key, 4, concurrent=False)
        await _settle("g" + "0" * 15)
        before = len(writes)
        assert (await _admissions(key, 20, concurrent=False)).count(429) == 20

        assert len(writes) == before

    async def test_slots_are_reserved_in_growing_batches(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Steady traffic costs far fewer writes than requests.

        Ref: stdapi/tenant_rate_limits.py:_Ledger.next_batch
        """
        writes: list[str] = []

        async def _spy(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401
            writes.append(str(args))
            return await add_to_item(*args, **kwargs)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _spy)
        key = await _declare_and_mint(
            limited_backend, key_id="h" + "0" * 15, requests_per_minute=640
        )

        statuses = await _admissions(key, 40, concurrent=False)
        await _settle("h" + "0" * 15)

        assert statuses.count(0) == 40
        # Batches of 1, 2, 4, 8, 16 and 32 cover the forty, plus one refill.
        assert len(writes) <= 8
        counter = await _counter("h" + "0" * 15)
        assert counter is not None
        requests, expires_at = (
            counter[REQUESTS_ATTRIBUTE],
            counter[EXPIRES_AT_ATTRIBUTE],
        )
        assert isinstance(requests, int)
        assert isinstance(expires_at, int)
        assert requests >= 40
        assert expires_at > time()

    async def test_a_request_the_token_limit_refuses_consumes_no_slot(
        self, limited_backend: SSMClient
    ) -> None:
        """A token refusal leaves the request slots the instance reserved intact.

        The token limit is tested before a slot is taken, so slots an
        instance paid a table write for stay with it; spending them on
        requests that were never served would refuse the tenant below its
        request limit for the rest of the minute.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        key_id = "a" + "1" * 15
        key = await _declare_and_mint(
            limited_backend, key_id, requests_per_minute=64, tokens_per_minute=100
        )
        # Fills the token limit, and teaches a mean no further request fits under.
        await _bill(key, 100)
        await _settle(key_id)

        assert await _admissions(key, 3, concurrent=False) == [429, 429, 429]
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        reserved = ledger.granted
        assert await _admissions(key, 3, concurrent=False) == [429, 429, 429]

        assert reserved > 0
        assert ledger.granted == reserved
        counter = await _counter(key_id)
        assert counter is not None
        # The one the billed request took, plus the ones still held here.
        assert counter[REQUESTS_ATTRIBUTE] == reserved + 1


class TestTokens:
    """Tokens are admitted on an estimate and reconciled from the billed usage.

    Ref: stdapi/tenant_rate_limits.py:debit_tenant_tokens
         stdapi/monitoring.py:_finalize_usage
         https://platform.claude.com/docs/en/api/rate-limits
    """

    async def test_billed_tokens_are_debited_after_the_response(
        self, limited_backend: SSMClient
    ) -> None:
        """Input, cache-write and output tokens count; cache reads are free.

        Ref: https://platform.claude.com/docs/en/api/rate-limits
        """
        key_id = "t" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=10000)
        request = _request_for()

        with log_request_event(request):
            await authenticate(credentials=None, x_api_key=key, request=request)
            record_bedrock_usage(
                "amazon.nova-micro-v1:0",
                input_tokens=100,
                cache_write_tokens=20,
                cached_tokens=50,
                output_tokens=30,
            )
        await _settle(key_id)

        counter = await _counter(key_id)
        assert counter is not None
        assert counter[TOKENS_ATTRIBUTE] == 150
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0

    async def test_the_token_limit_refuses_once_reached(
        self, limited_backend: SSMClient
    ) -> None:
        """A window whose tokens reached the limit refuses until the boundary.

        Ref: stdapi/tenant_rate_limits.py:_Ledger.projected_tokens
             stdapi/tenant_rate_limits.py:_refuse
        """
        key_id = "u" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=100)
        request = _request_for()

        with log_request_event(request):
            await authenticate(credentials=None, x_api_key=key, request=request)
            record_bedrock_usage(
                "amazon.nova-micro-v1:0", input_tokens=90, output_tokens=10
            )
        await _settle(key_id)
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        in_flight = ledger.inflight_estimate
        refused = _request_for()
        with pytest.raises(ApiError) as raised:
            await authenticate(credentials=None, x_api_key=key, request=refused)
        await settle_tenant_reservation(refused)

        assert raised.value.status == 429
        assert "100 tokens" in str(raised.value)
        assert 1 <= (tenant_retry_after(refused) or 0) <= WINDOW_SECONDS
        # A refusal took nothing, so its settlement gives nothing back.
        assert ledger.inflight_estimate == in_flight

    async def test_another_instances_tokens_are_learned_from_the_counter(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first request of a window reads the global total before admitting.

        Ref: stdapi/tenant_rate_limits.py:_ensure
             stdapi/tenant_rate_limits.py:_sync
        """
        key_id = "v" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=100)
        window = _current_window()
        await add_to_item(
            item_key("LIMIT", key_id),
            str(window),
            {TOKENS_ATTRIBUTE: 100},
            expires_at=window + 2 * WINDOW_SECONDS,
        )
        monkeypatch.setattr(tenant_rate_limits, "_LEDGERS", {})

        with pytest.raises(ApiError) as raised:
            await authenticate(credentials=None, x_api_key=key, request=_request_for())

        assert raised.value.status == 429

    async def test_the_estimate_carries_over_a_window_rollover(
        self, limited_backend: SSMClient, clock: _Clock
    ) -> None:
        """The mean tokens per request seen last minute is this minute's estimate.

        Admission counts the estimate of every request in flight, so the mean
        is what stops a burst from all being admitted against a fresh window.

        Ref: stdapi/tenant_rate_limits.py:_ledger_for
             stdapi/tenant_rate_limits.py:_Ledger.current_estimate
        """
        key_id = "w" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=1000)
        await _bill(key, 300, output_tokens=100)
        await _settle(key_id)
        previous = _current_window()
        clock.advance(WINDOW_SECONDS)

        # Three requests in flight fill the fresh window at 400 tokens each.
        statuses = await _admissions(key, 4, concurrent=False)

        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.window == previous + WINDOW_SECONDS
        assert ledger.estimate == 400
        assert statuses == [0, 0, 429, 429]

    async def test_tokens_billed_after_the_window_rolled_count_on_the_new_one(
        self, limited_backend: SSMClient, clock: _Clock
    ) -> None:
        """A response straddling the minute debits the minute it was billed in.

        A Realtime or stateful MCP session is admitted once and bills for as
        long as it lives; a ledger bound to the admission minute would take
        its tokens to a window nothing reads any more.

        Ref: stdapi/tenant_rate_limits.py:_current_ledger
        """
        key_id = "y" + "1" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=10000)
        admitted_in = _current_window()
        request = _request_for()

        with log_request_event(request):
            await authenticate(credentials=None, x_api_key=key, request=request)
            clock.advance(WINDOW_SECONDS)
            record_bedrock_usage(
                "amazon.nova-micro-v1:0", input_tokens=100, output_tokens=30
            )
        await _settle(key_id)

        current = await _counter(key_id)
        assert current is not None
        assert current[TOKENS_ATTRIBUTE] == 130
        previous = await _counter(key_id, admitted_in)
        assert previous is not None
        assert previous[TOKENS_ATTRIBUTE] == 0
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.window == admitted_in + WINDOW_SECONDS
        assert ledger.current_estimate() == 130
        assert ledger.inflight_estimate == 0

    async def test_a_window_billed_over_the_limit_admits_the_next_windows_first(
        self, limited_backend: SSMClient, clock: _Clock
    ) -> None:
        """A carried-over estimate above the limit cannot lock the key out.

        A refused request neither bills nor observes, so an estimate the
        admission test can never pass would be carried over for ever; it is
        capped at the limit instead, and the first request re-learns it.

        Ref: stdapi/tenant_rate_limits.py:_Ledger.admission_estimate
        """
        key_id = "y" + "2" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=400)
        await _bill(key, 500)
        await _settle(key_id)
        assert (await _admissions(key, 1, concurrent=False)) == [429]

        clock.advance(WINDOW_SECONDS)
        # Admitted, and billed: what it bills is what re-teaches the mean, and
        # a request left in flight would keep its share of the limit instead.
        await _bill(key, 100)
        await _settle(key_id)
        clock.advance(WINDOW_SECONDS)
        later = await _admissions(key, 1, concurrent=False)

        assert later == [0]

    async def test_a_key_never_billed_is_estimated_at_a_share_of_its_limit(
        self, limited_backend: SSMClient
    ) -> None:
        """A burst on a fresh instance is bounded before the first response.

        With no history the estimate would be zero and every request in
        flight would count for nothing, so a token-only limit would admit
        an unbounded burst.

        Ref: stdapi/tenant_rate_limits.py:_Ledger.admission_estimate
        """
        key = await _declare_and_mint(
            limited_backend, key_id="y" + "3" * 15, tokens_per_minute=8000
        )

        statuses = await _admissions(key, 12, concurrent=True)

        divisor = tenant_rate_limits._PRIOR_DIVISOR  # noqa: SLF001
        assert statuses.count(0) == divisor
        assert statuses.count(429) == 12 - divisor

    async def test_a_trained_down_estimate_cannot_admit_an_unbounded_burst(
        self, limited_backend: SSMClient
    ) -> None:
        """Requests in flight are bounded per key, whatever the mean was taught.

        A few cheap requests make the mean tiny, and the projection alone
        would then admit thousands of requests before any of them bills;
        the ceiling holds however small the estimate is, and a request limit
        is what raises it.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
        """
        key_id = "y" + "8" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=100000)
        await _bill(key, 20)
        await _settle(key_id)
        ceiling = tenant_rate_limits._INFLIGHT_MAX  # noqa: SLF001

        statuses = await _admissions(key, ceiling + 8, concurrent=True)

        assert statuses.count(0) == ceiling
        assert statuses.count(429) == 8

    async def test_the_in_flight_ceiling_is_not_reset_by_a_window_rollover(
        self, limited_backend: SSMClient, clock: _Clock
    ) -> None:
        """A request still unbilled keeps its place under the ceiling next minute.

        A Realtime session runs for minutes and bills nothing until its first
        turn. Handing the fresh window an empty in-flight count would sell the
        same 64 places again every minute, so a key could hold hundreds of
        sessions open on one instance against a ceiling of 64.

        Ref: stdapi/tenant_rate_limits.py:_ledger_for
             stdapi/tenant_rate_limits.py:_Reservation.release
        """
        key_id = "y" + "9" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=100000)
        await _bill(key, 20)
        await _settle(key_id)
        ceiling = tenant_rate_limits._INFLIGHT_MAX  # noqa: SLF001

        # Filled and never settled: every one of them is still in flight.
        assert (await _admissions(key, ceiling, concurrent=True)).count(0) == ceiling
        clock.advance(WINDOW_SECONDS)

        assert await _admissions(key, 1, concurrent=False) == [429]

    async def test_a_request_limit_lifts_the_in_flight_ceiling(
        self, limited_backend: SSMClient
    ) -> None:
        """A declared request limit replaces the 64 requests a token limit allows.

        The ceiling exists only because a mean taught down by cheap requests
        bounds nothing; a request limit is a bound the operator chose, so a
        burst wider than the ceiling is admitted against it instead.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        key_id = "y" + "a" * 15
        key = await _declare_and_mint(
            limited_backend, key_id, requests_per_minute=200, tokens_per_minute=100000
        )
        await _bill(key, 20)
        await _settle(key_id)
        ceiling = tenant_rate_limits._INFLIGHT_MAX  # noqa: SLF001

        statuses = await _admissions(key, ceiling + 8, concurrent=True)

        assert statuses.count(0) == ceiling + 8
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_requests == ceiling + 8

    async def test_a_request_draining_usage_several_times_is_one_request(
        self, limited_backend: SSMClient
    ) -> None:
        """Every turn of a Realtime session reaches the counter, as one request.

        The session drains its usage on every turn, under the WebSocket's own
        request scope; counting each drain as a request would collapse the
        mean to one turn's tokens and admit far more sessions than the limit
        holds.

        Ref: stdapi/realtime.py:RealtimeSession._record_usage
             stdapi/tenant_rate_limits.py:_Reservation.debit
        """
        key_id = "k" + "3" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=100000)

        async def _unused(*_: Any) -> Any:  # noqa: ANN401
            pytest.fail("the transport is never used")

        websocket = WebSocket(_websocket_scope(key), _unused, _unused)
        with log_request_event(websocket):
            await verify_websocket_credentials(key, websocket.scope)
            for turn in (200, 300, 400):
                record_bedrock_usage("amazon.nova-micro-v1:0", input_tokens=turn)
                flush_usage_log_event(0)
        await _settle(key_id)

        counter = await _counter(key_id)
        assert counter is not None
        assert counter[TOKENS_ATTRIBUTE] == 900
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.observed_requests == 1
        assert ledger.current_estimate() == 900
        assert ledger.inflight_estimate == 0

    async def test_requests_billing_nothing_do_not_lower_the_estimate(
        self, limited_backend: SSMClient
    ) -> None:
        """Only a billed request feeds the mean a request is admitted on.

        Interleaving refused bodies or empty calls would otherwise drive the
        mean to zero and switch the token limit off.

        Ref: stdapi/tenant_rate_limits.py:_Reservation.debit
        """
        key_id = "y" + "4" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=10000)
        await _bill(key, 600)
        for _ in range(5):
            request = _request_for()
            await authenticate(credentials=None, x_api_key=key, request=request)
            await settle_tenant_reservation(request)

        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.observed_requests == 1
        assert ledger.current_estimate() == 600
        assert ledger.inflight_estimate == 0

    async def test_a_settled_batch_is_not_debited_to_its_reader(
        self, limited_backend: SSMClient
    ) -> None:
        """Batch usage is recorded by whichever request reads the batch first.

        That request may be another tenant's, and the tokens were billed
        outside any request: they never count against the reader's limit.

        Ref: stdapi/batches.py:settle
        """
        key_id = "y" + "5" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=1000)

        request = await _bill(
            key, 5000000, output_tokens=2000000, tier="batch", region="us-east-1"
        )
        await settle_tenant_reservation(request)
        await _settle(key_id)

        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.observed_tokens == 0
        assert ledger.inflight_estimate == 0
        counter = await _counter(key_id)
        assert counter is not None
        assert counter.get(TOKENS_ATTRIBUTE, 0) == 0
        assert (await _admissions(key, 1, concurrent=False)) == [0]

    async def test_tokens_a_failed_flush_carried_are_written_by_the_next(
        self, limited_backend: SSMClient, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled table loses no billed tokens: the flush keeps them pending.

        Ref: stdapi/tenant_rate_limits.py:_sync
        """
        key_id = "y" + "6" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=10000)
        await _bill(key, 100)
        TestTableUnavailable._break_counter(monkeypatch)  # noqa: SLF001
        await _settle(key_id)

        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.unflushed_tokens == 100
        assert ledger.failure is not None

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", add_to_item)
        clock.advance(2)
        await _bill(key, 50)
        await _settle(key_id)

        counter = await _counter(key_id)
        assert counter is not None
        assert counter[TOKENS_ATTRIBUTE] == 150
        assert ledger.unflushed_tokens == 0

    async def test_a_streamed_response_debits_in_the_stream_scope(
        self, limited_backend: SSMClient
    ) -> None:
        """Usage arriving in a trailing event still reaches the tenant's counter.

        Ref: stdapi/monitoring.py:_rebuild_and_log_stream
        """
        key_id = "x" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=10000)
        request = _request_for()

        async def _model_stream() -> AsyncGenerator[bytes]:
            yield b"first"
            yield b"second"
            record_bedrock_usage(
                "amazon.nova-micro-v1:0", input_tokens=7, output_tokens=5
            )

        # Consumed inside the request context, as the body of a real response
        # is: the stream's own log scope reads the request's identifiers.
        with log_request_event(request):
            await authenticate(credentials=None, x_api_key=key, request=request)
            stream = await log_request_stream_event(_model_stream())
            chunks = [chunk async for chunk in stream]
        await _settle(key_id)

        assert chunks == [b"first", b"second"]
        counter = await _counter(key_id)
        assert counter is not None
        assert counter[TOKENS_ATTRIBUTE] == 12


class TestLedgers:
    """The per-instance ledger registry stays cheap on the request path.

    Ref: stdapi/tenant_rate_limits.py:_ledger_for
    """

    def test_past_windows_are_swept_once_per_window(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full registry is scanned once a minute, not once per new key.

        With more keys active in a minute than the registry holds, every
        first request of a key it has not seen would otherwise scan the
        whole registry and find nothing to drop.

        Ref: stdapi/tenant_rate_limits.py:_ledger_for
             stdapi/tenant_rate_limits.py:_LEDGERS_MAX
        """

        class _Counting(dict[str, Any]):
            scans = 0

            def items(self) -> Any:  # noqa: ANN401
                type(self).scans += 1
                return super().items()

        ledgers = _Counting()
        monkeypatch.setattr(tenant_rate_limits, "_LEDGERS", ledgers)
        monkeypatch.setattr(tenant_rate_limits, "_SWEPT_WINDOW", 0)
        capacity = tenant_rate_limits._LEDGERS_MAX  # noqa: SLF001
        window = _current_window()
        past = window - WINDOW_SECONDS
        for i in range(capacity):
            ledgers[f"p{i}"] = tenant_rate_limits._Ledger(f"p{i}", past, None, 100)  # noqa: SLF001

        tenant_rate_limits._ledger_for("n1", None, 100)  # noqa: SLF001
        assert ledgers.scans == 1
        assert set(ledgers) == {"n1"}

        for i in range(capacity):
            ledgers[f"c{i}"] = tenant_rate_limits._Ledger(f"c{i}", window, None, 100)  # noqa: SLF001
        tenant_rate_limits._ledger_for("n2", None, 100)  # noqa: SLF001
        tenant_rate_limits._ledger_for("n3", None, 100)  # noqa: SLF001
        assert ledgers.scans == 1

        clock.advance(WINDOW_SECONDS)
        tenant_rate_limits._ledger_for("n4", None, 100)  # noqa: SLF001
        assert ledgers.scans == 2
        assert set(ledgers) == {"n4"}


class TestPublishedNumbers:
    """The shares and the ceiling the documentation states as plain numbers.

    Every other test reads them back from the module, so a value change would
    pass them all while silently rewriting what six documentation places
    promise an operator sizing a limit.

    Ref: docs/operations_authentication_security.md#tenant-rate-limits
         docs/operations_configuration_authentication.md#tenant-rate-limit-tokens-per-minute
         docs/operations_troubleshooting.md
    """

    def test_the_documented_shares_and_ceiling_are_the_ones_in_force(self) -> None:
        """An eighth of each limit, and 64 requests in flight per instance.

        The pages state them as figures an operator sizes a limit against:
        "an eighth of its token limit per request in flight", "about eight
        concurrent requests", "batches of up to an eighth of the limit" and
        "at 64 per instance".

        Ref: stdapi/tenant_rate_limits.py:_PRIOR_DIVISOR
             stdapi/tenant_rate_limits.py:_BATCH_DIVISOR
             stdapi/tenant_rate_limits.py:_INFLIGHT_MAX
        """
        assert tenant_rate_limits._PRIOR_DIVISOR == 8  # noqa: SLF001
        assert tenant_rate_limits._BATCH_DIVISOR == 8  # noqa: SLF001
        assert tenant_rate_limits._INFLIGHT_MAX == 64  # noqa: SLF001


class TestRotation:
    """A rotated key keeps its key ID, so both secrets share one budget.

    Ref: stdapi/tenant_keys.py:_matches
    """

    async def test_both_secrets_of_a_rotated_key_share_one_counter(
        self, limited_backend: SSMClient
    ) -> None:
        """The superseded secret and the current one count on the same item.

        Ref: stdapi/tenant_keys.py:_matches
             stdapi/tenant_rate_limits.py:_ledger_for
        """
        key_id = "r" + "0" * 15
        current = await _declare_and_mint(
            limited_backend, key_id, requests_per_minute=3
        )
        secret_item = await get_item("TENANT", f"secret#{key_id}")
        assert secret_item is not None
        previous_salt = token_bytes(16)
        previous_secret = "p" * 43
        await put_item(
            {
                **secret_item,
                "previous_secret_hash": tenant_keys._hash_secret(  # noqa: SLF001
                    previous_secret, previous_salt
                ),
                "previous_salt": previous_salt,
                "previous_until": int(time()) + 3600,
            }
        )
        superseded = f"{KEY_PREFIX}{key_id}-{previous_secret}"

        statuses = [
            *(await _admissions(current, 2, concurrent=False)),
            *(await _admissions(superseded, 2, concurrent=False)),
        ]

        assert statuses == [0, 0, 0, 429]


class TestTableUnavailable:
    """The counter failing closes the door, once the granted slots are used.

    Ref: stdapi/tenant_rate_limits.py:_sync
         stdapi/aws_dynamodb.py:_failure
    """

    @staticmethod
    def _break_counter(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make every counter write fail as a denied UpdateItem."""

        async def _denied(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401, ARG001
            msg = (
                "the server role is missing the IAM permission dynamodb:UpdateItem "
                "on the table"
            )
            raise TableUnavailableError(msg)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _denied)

    async def test_granted_slots_serve_then_the_refusal_is_a_503(
        self,
        limited_backend: SSMClient,
        request_log: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Slots already granted keep serving; an empty ledger fails closed.

        A 429 would tell the client it is over its limit, which is false; the
        503 is what both SDKs retry, and the operator reads the IAM action.

        Ref: stdapi/tenant_rate_limits.py:_ensure
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        key_id = "s" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=64)
        # The second request reserves a batch of two and uses one of them.
        await _admissions(key, 2, concurrent=False)
        await _settle(key_id)
        granted = tenant_rate_limits._LEDGERS[key_id].granted  # noqa: SLF001
        assert granted >= 1
        self._break_counter(monkeypatch)

        statuses = await _admissions(key, granted + 1, concurrent=False)

        assert statuses[:granted] == [0] * granted
        assert statuses[-1] == 503
        assert "dynamodb:UpdateItem" in str(request_log["error_detail"])
        assert request_log["level"] == "warning"

    async def test_the_refusal_is_the_generic_feature_message(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller reads nothing of the table, the action or the region.

        Ref: stdapi/api_errors.py:FeatureUnavailableError
        """
        key = await _declare_and_mint(
            limited_backend, key_id="y" + "0" * 15, requests_per_minute=8
        )
        self._break_counter(monkeypatch)

        with pytest.raises(FeatureUnavailableError) as raised:
            await authenticate(credentials=None, x_api_key=key, request=_request_for())

        assert raised.value.status == 503
        assert "Tenant rate limiting is not available" in str(raised.value)
        assert "dynamodb" not in str(raised.value).lower()

    async def test_a_failed_write_is_not_retried_on_every_request(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled table is not hammered by the refusals it causes.

        Ref: stdapi/tenant_rate_limits.py:_ensure
             stdapi/tenant_rate_limits.py:_RETRY_SECONDS
        """
        attempts = 0

        async def _denied(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401, ARG001
            nonlocal attempts
            attempts += 1
            msg = "dynamodb:UpdateItem failed (Throttling)"
            raise TableUnavailableError(msg)

        key = await _declare_and_mint(
            limited_backend, key_id="z" + "0" * 15, requests_per_minute=8
        )
        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _denied)

        statuses = await _admissions(key, 10, concurrent=False)

        assert statuses == [503] * 10
        assert attempts == 1

    async def test_a_stalled_write_ends_in_the_fail_closed_refusal(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A table that hangs before giving up refuses with the 503, never a 429.

        The documented promise is that a stalling table is given up on and the
        request refused, for the one waiting on the write and for every
        request of that key queued behind it -- neither left waiting for ever
        nor admitted because the counter could not say no.

        Ref: stdapi/tenant_rate_limits.py:_sync_now
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        key_id = "s" + "1" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=64)
        reached, released = Event(), Event()

        async def _stalled(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401, ARG001
            reached.set()
            await released.wait()
            msg = "dynamodb:UpdateItem timed out on the table"
            raise TableUnavailableError(msg)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _stalled)
        waiting = create_task(_admissions(key, 1, concurrent=False))
        await reached.wait()
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        queued = create_task(_admissions(key, 1, concurrent=False))
        for _ in range(100):
            if ledger.waiting == 2:
                break
            await sleep(0)

        assert ledger.waiting == 2, "the second request joined the write in flight"
        assert not waiting.done()
        released.set()
        assert await waiting == [503]
        assert await queued == [503]

    async def test_the_fail_closed_refusal_leaves_the_ledger_untouched(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settling a 503 gives back nothing, since nothing was taken.

        The reservation is on the scope before the counter is written, so the
        response's settlement finds it; releasing an estimate never added
        would drive the in-flight total negative and admit past the limit
        once the table recovers.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
        """
        key_id = "y" + "7" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=8000)
        self._break_counter(monkeypatch)
        request = _request_for()

        with pytest.raises(FeatureUnavailableError):
            await authenticate(credentials=None, x_api_key=key, request=request)
        await settle_tenant_reservation(request)

        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0
        assert ledger.observed_requests == 0


class TestHeaders:
    """Every limited response carries the vendors' own rate-limit headers.

    Ref: stdapi/main.py:set_rate_limit_headers
         https://developers.openai.com/api/docs/guides/rate-limits
         https://platform.claude.com/docs/en/api/rate-limits
    """

    async def test_openai_routes_carry_the_x_ratelimit_headers(
        self, limited_backend: SSMClient
    ) -> None:
        """Six headers, resets as Go durations, on an ordinary response.

        Both limits are declared, which is what publishes all six: each triple
        is emitted only for the limit it counts.

        Ref: stdapi/tenant_rate_limits.py:_openai_headers
             https://developers.openai.com/api/docs/guides/rate-limits
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key = await _declare_and_mint(
            limited_backend,
            key_id="o" + "0" * 15,
            requests_per_minute=60,
            tokens_per_minute=150000,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/chat/completions", json={}, headers={"x-api-key": key}
            )

        # Refused by the body schema, once the limiter admitted it.
        assert response.status_code == 400
        assert response.headers["x-ratelimit-limit-requests"] == "60"
        assert response.headers["x-ratelimit-remaining-requests"] == "59"
        assert _GO_DURATION.match(response.headers["x-ratelimit-reset-requests"])
        assert response.headers["x-ratelimit-limit-tokens"] == "150000"
        # The request's own estimate is in flight while the headers are built.
        assert response.headers["x-ratelimit-remaining-tokens"] == str(
            150000 - 150000 // tenant_rate_limits._PRIOR_DIVISOR  # noqa: SLF001
        )
        assert _GO_DURATION.match(response.headers["x-ratelimit-reset-tokens"])
        assert "retry-after" not in response.headers

    async def test_the_remaining_tokens_follow_what_was_billed(
        self, limited_backend: SSMClient
    ) -> None:
        """A key that billed tokens reads fewer remaining than its limit.

        Ref: stdapi/tenant_rate_limits.py:_Ledger.remaining_tokens
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key_id = "o" + "1" * 15
        key = await _declare_and_mint(
            limited_backend, key_id, requests_per_minute=60, tokens_per_minute=150000
        )
        await _bill(key, 100, output_tokens=50)
        await _settle(key_id)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            openai = await client.post(
                "/v1/chat/completions", json={}, headers={"x-api-key": key}
            )
            anthropic = await client.post(
                f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
                json={},
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )

        # The 150 billed, plus this request's own estimate: the mean, 150.
        assert openai.headers["x-ratelimit-remaining-tokens"] == str(150000 - 300)
        assert anthropic.headers["anthropic-ratelimit-tokens-remaining"] == str(
            150000 - 300
        )
        assert int(openai.headers["x-ratelimit-remaining-tokens"]) < 150000

    async def test_anthropic_routes_carry_the_anthropic_ratelimit_headers(
        self, limited_backend: SSMClient
    ) -> None:
        """The Anthropic names, resets in RFC 3339.

        Ref: stdapi/tenant_rate_limits.py:_anthropic_headers
             https://platform.claude.com/docs/en/api/rate-limits
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key = await _declare_and_mint(
            limited_backend,
            key_id="n" + "0" * 15,
            requests_per_minute=60,
            tokens_per_minute=150000,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
                json={},
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )

        assert response.status_code == 400
        assert response.headers["anthropic-ratelimit-requests-limit"] == "60"
        assert response.headers["anthropic-ratelimit-requests-remaining"] == "59"
        reset = response.headers["anthropic-ratelimit-requests-reset"]
        assert _RFC3339.match(reset)
        assert datetime.fromisoformat(reset) > datetime.now(UTC)
        assert response.headers["anthropic-ratelimit-tokens-limit"] == "150000"
        assert response.headers["anthropic-ratelimit-tokens-remaining"] == str(
            150000 - 150000 // tenant_rate_limits._PRIOR_DIVISOR  # noqa: SLF001
        )
        assert _RFC3339.match(response.headers["anthropic-ratelimit-tokens-reset"])
        assert "x-ratelimit-limit-requests" not in response.headers

    async def test_a_refusal_carries_retry_after_on_both_dialects(
        self, limited_backend: SSMClient
    ) -> None:
        """The 429 is each vendor's rate_limit_error, with a retry delay in seconds.

        Ref: stdapi/main.py:set_retry_after_header
             stdapi/api_errors.py:RateLimitExceededError
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key = await _declare_and_mint(
            limited_backend, key_id="p" + "0" * 15, requests_per_minute=1
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            await client.post(
                "/v1/chat/completions", json={}, headers={"x-api-key": key}
            )
            openai = await client.post(
                "/v1/chat/completions",
                json={"stream": True},
                headers={"x-api-key": key},
            )
            anthropic = await client.post(
                f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
                json={},
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )

        assert openai.status_code == 429
        assert openai.headers["content-type"].startswith("application/json")
        assert openai.json()["error"]["type"] == "rate_limit_error"
        assert openai.json()["error"]["code"] == "rate_limit_exceeded"
        assert 1 <= int(openai.headers["retry-after"]) <= WINDOW_SECONDS
        assert openai.headers["x-ratelimit-remaining-requests"] == "0"
        assert anthropic.status_code == 429
        assert anthropic.json()["error"]["type"] == "rate_limit_error"
        assert 1 <= int(anthropic.headers["retry-after"]) <= WINDOW_SECONDS
        assert anthropic.headers["anthropic-ratelimit-requests-remaining"] == "0"

    async def test_each_triple_is_published_only_for_the_limit_it_counts(
        self, limited_backend: SSMClient
    ) -> None:
        """A tenant capped on one of the two carries that triple and no other.

        Publishing a `limit`/`remaining`/`reset` triple for a limit nobody
        declared would answer a number that counts nothing, which a client
        sizing its own backoff reads as a budget.

        Ref: stdapi/tenant_rate_limits.py:_openai_headers
             stdapi/tenant_rate_limits.py:_anthropic_headers
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        tokens_only = await _declare_and_mint(
            limited_backend, key_id="o" + "6" * 15, tokens_per_minute=150000
        )
        requests_only = await _declare_and_mint(
            limited_backend, key_id="n" + "1" * 15, requests_per_minute=60
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            openai = await client.post(
                "/v1/chat/completions", json={}, headers={"x-api-key": tokens_only}
            )
            anthropic = await client.post(
                f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
                json={},
                headers={"x-api-key": requests_only, "anthropic-version": "2023-06-01"},
            )

        assert openai.headers["x-ratelimit-limit-tokens"] == "150000"
        assert "x-ratelimit-limit-requests" not in openai.headers
        assert "x-ratelimit-remaining-requests" not in openai.headers
        assert "x-ratelimit-reset-requests" not in openai.headers
        assert anthropic.headers["anthropic-ratelimit-requests-limit"] == "60"
        assert "anthropic-ratelimit-tokens-limit" not in anthropic.headers
        assert "anthropic-ratelimit-tokens-remaining" not in anthropic.headers
        assert "anthropic-ratelimit-tokens-reset" not in anthropic.headers

    async def test_the_other_dialects_refuse_with_a_plain_429(
        self, limited_backend: SSMClient
    ) -> None:
        """Cohere and Ollama answer their own envelope, a retry delay and no headers.

        Neither vendor publishes rate-limit headers or an error type, so the
        refusal is that dialect's plain error body with the status and the
        delay: inventing either would publish a contract the vendor has not.

        Ref: stdapi/api_providers/cohere.py:_format_error
             stdapi/api_providers/ollama.py:_format_error
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key_id = "p" + "1" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=1)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            admitted = await client.post(
                f"{SETTINGS.ollama_routes_prefix}/api/chat",
                json={},
                headers={"x-api-key": key},
            )
            ollama = await client.post(
                f"{SETTINGS.ollama_routes_prefix}/api/chat",
                json={},
                headers={"x-api-key": key},
            )
            cohere = await client.post(
                f"{SETTINGS.cohere_routes_prefix}/v2/rerank",
                json={},
                headers={"x-api-key": key},
            )

        # Refused by the body schema, once the limiter admitted it.
        assert admitted.status_code == 400
        assert not [name for name in admitted.headers if "ratelimit" in name]
        for refused in (ollama, cohere):
            assert refused.status_code == 429
            assert 1 <= int(refused.headers["retry-after"]) <= WINDOW_SECONDS
            assert not [name for name in refused.headers if "ratelimit" in name]
        assert key_id in ollama.json()["error"]
        assert key_id in cohere.json()["message"]
        assert "error" not in cohere.json()

    async def test_an_unreachable_counter_is_a_529_on_anthropic_routes(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fail-closed 503 takes the Anthropic dialect's overloaded shape.

        Ref: stdapi/api_providers/anthropic.py:_format_error
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key = await _declare_and_mint(
            limited_backend, key_id="q" + "0" * 15, requests_per_minute=8
        )
        TestTableUnavailable._break_counter(monkeypatch)  # noqa: SLF001

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
                json={},
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )

        assert response.status_code == 529
        assert response.json()["error"]["type"] == "overloaded_error"
        assert "retry-after" not in response.headers


class TestSettlement:
    """The in-flight estimate is given back on every way a request can end.

    Ref: stdapi/tenant_rate_limits.py:settle_tenant_reservation
         stdapi/main.py:_middleware
    """

    async def test_a_response_billing_nothing_releases_its_estimate(
        self, limited_backend: SSMClient
    ) -> None:
        """A refused body bills no token, so only the response's settlement releases.

        Without it every such request would hold its estimate for the rest of
        the minute and wedge the token limit.

        Ref: stdapi/tenant_rate_limits.py:settle_tenant_reservation
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        key_id = "o" + "2" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=8000)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/chat/completions", json={}, headers={"x-api-key": key}
            )

        assert response.status_code == 400
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0
        assert ledger.observed_requests == 0

    @staticmethod
    async def _run_middleware(key: str, failure: Exception) -> int | None:
        """Drive the middleware over a disconnected client whose handler raises.

        Args:
            key: The tenant key the handler authenticates with.
            failure: What the handler raises once admitted.

        Returns:
            The status answered, or None when the failure escaped unanswered.
        """
        from stdapi.main import _middleware  # noqa: PLC0415

        async def _disconnected() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        request = Request(_scope("/v1/chat/completions"), receive=_disconnected)

        async def _handler(request: Request) -> Any:  # noqa: ANN401
            await authenticate(credentials=None, x_api_key=key, request=request)
            raise failure

        try:
            return (await _middleware(request, _handler)).status_code
        except type(failure):
            return None

    async def test_a_client_that_left_releases_its_estimate(
        self, limited_backend: SSMClient
    ) -> None:
        """The 499 answered to a vanished client settles what it reserved.

        Ref: stdapi/main.py:_middleware
        """
        key_id = "o" + "3" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=8000)

        status = await self._run_middleware(key, RuntimeError("No response returned."))

        assert status == 499
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0

    async def test_an_unhandled_failure_releases_its_estimate(
        self, limited_backend: SSMClient
    ) -> None:
        """A request that sends no response at all still settles what it reserved.

        Ref: stdapi/main.py:_middleware
        """
        key_id = "o" + "4" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=8000)

        status = await self._run_middleware(key, ValueError("boom"))

        assert status is None
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0

    async def test_a_request_cancelled_during_the_counter_write_took_nothing(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client leaving while the minute's first write is in flight takes nothing.

        The reservation is on the scope before the write, so the middleware
        settles it on the cancellation; giving back an estimate that was
        never taken would drive the in-flight total negative and admit past
        the limit for the rest of the minute.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
        """
        key_id = "o" + "5" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=8000)
        reached, released = Event(), Event()

        async def _stalled(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401
            reached.set()
            await released.wait()
            return await add_to_item(*args, **kwargs)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _stalled)
        request = _request_for()
        waiter = create_task(
            authenticate(credentials=None, x_api_key=key, request=request)
        )
        # The write runs on its own task: the waiter is suspended on it by now.
        await reached.wait()
        waiter.cancel()
        with pytest.raises(CancelledError):
            await waiter
        await settle_tenant_reservation(request)
        released.set()
        await _settle(key_id)

        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0
        assert ledger.inflight_requests == 0


class TestWebSocketAndMcp:
    """Where a request is counted once, and where it must not be counted twice.

    Ref: stdapi/auth.py:verify_websocket_credentials
         stdapi/tenant_keys.py:resume_tenant
         stdapi/mcp.py:_inject_request_id
    """

    async def test_the_handshake_counts_one_request(
        self, limited_backend: SSMClient
    ) -> None:
        """A WebSocket handshake takes one slot; its scope carries the reservation.

        Ref: stdapi/auth.py:verify_websocket_credentials
        """
        key_id = "k" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=2)
        scope: dict[str, Any] = {
            "type": "websocket",
            "headers": [],
            "route": SimpleNamespace(path_format="/v1/realtime", tags=[]),
        }

        await verify_websocket_credentials(key, scope)
        await verify_websocket_credentials(key, dict(scope, state={}))
        with pytest.raises(ApiError) as raised:
            await verify_websocket_credentials(key, dict(scope, state={}))

        assert STATE_KEY in scope["state"]
        assert raised.value.status == 429

    async def test_a_handshake_ending_before_a_turn_gives_back_its_estimate(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connection refused after admission, or closed unbilled, holds nothing.

        Nothing on a WebSocket bills a token before the first turn, so the
        connection's end is the only thing that can release what the
        handshake reserved; eight such connections would otherwise wedge a
        fresh key at its token limit for the rest of the minute.

        Ref: stdapi/realtime.py:serve_realtime_session
        """
        key_id = "k" + "1" * 15
        key = await _declare_and_mint(limited_backend, key_id, tokens_per_minute=8000)

        scope, sent = await _handshake(key, monkeypatch)

        assert STATE_KEY in scope["state"]
        assert sent[-1]["type"] == "websocket.close"
        ledger = tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001
        assert ledger.inflight_estimate == 0
        assert ledger.inflight_requests == 0

    async def test_a_refused_handshake_answers_an_error_event_and_close_3000(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The upgrade succeeds, then one terminal error event and the close frame.

        Ref: stdapi/realtime.py:_refuse_events
             docs/operations_authentication_security.md
        """
        key = await _declare_and_mint(
            limited_backend, key_id="k" + "2" * 15, requests_per_minute=1
        )

        await _handshake(key, monkeypatch)
        _, sent = await _handshake(key, monkeypatch)

        assert [message["type"] for message in sent] == [
            "websocket.accept",
            "websocket.send",
            "websocket.close",
        ]
        event = json.loads(sent[1]["text"])
        assert event["type"] == "error"
        assert event["error"]["type"] == "invalid_request_error"
        assert event["error"]["code"] == "rate_limit_exceeded"
        assert "1 request per minute" in event["error"]["message"]
        assert sent[2]["code"] == 3000
        assert sent[2]["reason"] == "invalid_request_error.rate_limit_exceeded"

    async def test_an_unwritable_counter_closes_the_handshake_as_a_server_error(
        self, limited_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A counter that cannot be written ends the session as a server error.

        The upgrade has already succeeded, so the fail-closed refusal reaches
        the client as a terminal event rather than a status. It is the
        deployment's failure, not the tenant's: `server_error`, where a limit
        refusal on the same route sends `invalid_request_error`.

        Ref: stdapi/realtime.py:_refuse
             docs/operations_authentication_security.md#tenant-rate-limits
        """
        key = await _declare_and_mint(
            limited_backend, key_id="k" + "4" * 15, requests_per_minute=8
        )
        TestTableUnavailable._break_counter(monkeypatch)  # noqa: SLF001

        _, sent = await _handshake(key, monkeypatch)

        assert [message["type"] for message in sent] == [
            "websocket.accept",
            "websocket.send",
            "websocket.close",
        ]
        event = json.loads(sent[1]["text"])
        assert event["type"] == "error"
        assert event["error"]["type"] == "server_error"
        assert event["error"]["code"] == "feature_unavailable"
        assert "dynamodb" not in event["error"]["message"].lower()
        assert sent[2]["code"] == 3000
        assert sent[2]["reason"] == "server_error.feature_unavailable"

    async def test_a_resumed_realtime_grant_counts_one_request(
        self, limited_backend: SSMClient
    ) -> None:
        """A session opened with a minted client secret counts at the handshake.

        Ref: stdapi/tenant_keys.py:resume_tenant
        """
        key_id = "j" + "0" * 15
        await _declare_and_mint(limited_backend, key_id, requests_per_minute=1)
        request = _request_for("/v1/realtime")
        token = REQUEST.set(request)
        try:
            tenant = await resume_tenant(key_id)
            with pytest.raises(ApiError) as raised:
                await resume_tenant(key_id)
        finally:
            REQUEST.reset(token)

        assert tenant.requests_per_minute == 1
        assert STATE_KEY in request.scope["state"]
        assert raised.value.status == 429

    async def test_the_internal_mcp_leg_is_not_counted_again(
        self, limited_backend: SSMClient
    ) -> None:
        """The tool call's API leg reuses the MCP request's reservation.

        The leg is recognized by the MCP user agent and the internal request-id
        header, which no external client can forge together.

        Ref: stdapi/tenant_rate_limits.py:_is_internal_leg
             stdapi/mcp.py:_inject_request_id
        """
        key_id = "i" + "0" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=1)
        outer = _request_for("/mcp")
        inner = Request(
            _scope(
                "/v1/chat/completions",
                {"user-agent": MCP_USER_AGENT, INTERNAL_REQUEST_ID_HEADER: "abc"},
            )
        )

        await authenticate(credentials=None, x_api_key=key, request=outer)
        await authenticate(credentials=None, x_api_key=key, request=inner)

        assert inner.scope["state"][STATE_KEY] is outer.scope["state"][STATE_KEY]
        with pytest.raises(ApiError):
            await authenticate(credentials=None, x_api_key=key, request=_request_for())

    async def test_the_internal_leg_of_another_tenant_is_counted_on_its_own(
        self, limited_backend: SSMClient
    ) -> None:
        """A leg dispatched under another tenant's reservation takes its own.

        A stateful MCP session is served on the task that opened it, so a
        tool call presenting a leaked session ID would otherwise bill the
        first tenant.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
        """
        first = await _declare_and_mint(
            limited_backend, key_id="i" + "1" * 15, requests_per_minute=1
        )
        second_id = "i" + "2" * 15
        second = await _declare_and_mint(
            limited_backend, second_id, requests_per_minute=1
        )
        outer = _request_for("/mcp")
        inner = Request(
            _scope(
                "/v1/chat/completions",
                {"user-agent": MCP_USER_AGENT, INTERNAL_REQUEST_ID_HEADER: "abc"},
            )
        )

        await authenticate(credentials=None, x_api_key=first, request=outer)
        await authenticate(credentials=None, x_api_key=second, request=inner)

        reservation = inner.scope["state"][STATE_KEY]
        assert reservation is not outer.scope["state"][STATE_KEY]
        assert reservation.ledger.key_id == second_id
        with pytest.raises(ApiError) as raised:
            await authenticate(
                credentials=None, x_api_key=second, request=_request_for()
            )
        assert raised.value.status == 429

    async def test_the_internal_leg_of_a_past_window_is_counted_afresh(
        self, limited_backend: SSMClient, clock: _Clock
    ) -> None:
        """A session's tool call after the minute rolled reserves on the new one.

        Ref: stdapi/tenant_rate_limits.py:admit_tenant_request
        """
        key_id = "i" + "3" * 15
        key = await _declare_and_mint(limited_backend, key_id, requests_per_minute=2)
        outer = _request_for("/mcp")
        inner = Request(
            _scope(
                "/v1/chat/completions",
                {"user-agent": MCP_USER_AGENT, INTERNAL_REQUEST_ID_HEADER: "abc"},
            )
        )

        await authenticate(credentials=None, x_api_key=key, request=outer)
        clock.advance(WINDOW_SECONDS)
        await authenticate(credentials=None, x_api_key=key, request=inner)

        reservation = inner.scope["state"][STATE_KEY]
        assert reservation is not outer.scope["state"][STATE_KEY]
        assert reservation.ledger.window == _current_window()
        assert reservation.ledger is tenant_rate_limits._LEDGERS[key_id]  # noqa: SLF001

    async def test_the_mcp_markers_alone_do_not_skip_the_count(
        self, limited_backend: SSMClient
    ) -> None:
        """A client sending only the user agent is counted like any other.

        Ref: stdapi/tenant_rate_limits.py:_is_internal_leg
        """
        key = await _declare_and_mint(
            limited_backend, key_id="l" + "0" * 15, requests_per_minute=1
        )
        forged = Request(_scope("/v1/chat/completions", {"user-agent": MCP_USER_AGENT}))

        await authenticate(credentials=None, x_api_key=key, request=forged)
        with pytest.raises(ApiError) as raised:
            await authenticate(credentials=None, x_api_key=key, request=_request_for())

        assert raised.value.status == 429


class TestCounterWrites:
    """The atomic counter write the limiter is built on, and its startup probe.

    Ref: stdapi/aws_dynamodb.py:add_to_item
         https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
    """

    async def test_an_add_returns_the_running_totals(self, dynamodb_table: str) -> None:
        """Every write answers with the totals after it, which is the global view.

        Ref: stdapi/aws_dynamodb.py:add_to_item
        """
        first = await add_to_item(
            "LIMIT#k", "1", {"requests": 3, "tokens": 10}, expires_at=99
        )
        second = await add_to_item(
            "LIMIT#k", "1", {"requests": 2, "tokens": 5}, expires_at=42
        )

        assert first["requests"] == 3
        assert first["tokens"] == 10
        assert second["requests"] == 5
        assert second["tokens"] == 15
        item = await get_item("LIMIT#k", "1")
        assert item is not None
        assert item[EXPIRES_AT_ATTRIBUTE] == 99, "the first writer's expiry is kept"

    async def test_a_negative_add_releases(self, dynamodb_table: str) -> None:
        """A grant can be given back, which is what a release would be.

        Ref: stdapi/aws_dynamodb.py:add_to_item
        """
        await add_to_item("LIMIT#r", "1", {"requests": 4}, expires_at=99)

        totals = await add_to_item("LIMIT#r", "1", {"requests": -1}, expires_at=99)

        assert totals["requests"] == 3

    async def test_a_failure_names_the_update_action(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator reads dynamodb:UpdateItem, the table and the region.

        Ref: stdapi/aws_dynamodb.py:_failure
        """
        from tests._helpers import make_client_error  # noqa: PLC0415

        denial = make_client_error(
            "AccessDeniedException",
            "UpdateItem",
            message=(
                "User: arn:aws:sts::123456789012:assumed-role/x/y is not "
                "authorized to perform: dynamodb:UpdateItem on resource: "
                f"arn:aws:dynamodb:us-east-1:123456789012:table/{dynamodb_table}"
            ),
        )

        class _Client:
            async def update_item(self, **_: Any) -> None:  # noqa: ANN401
                raise denial

        monkeypatch.setattr("stdapi.aws_dynamodb._client", _Client)

        with pytest.raises(TableUnavailableError) as raised:
            await add_to_item("LIMIT#d", "1", {"requests": 1}, expires_at=99)

        assert "dynamodb:UpdateItem" in raised.value.detail
        assert dynamodb_table in raised.value.detail

    async def test_the_startup_probe_reports_a_denied_write(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment declaring limits learns at startup that it cannot count.

        Ref: stdapi/tenant_rate_limits.py:verify_counter_access
        """
        from tests._helpers import make_event_log  # noqa: PLC0415

        start_event = make_event_log(type="start")
        await verify_counter_access(start_event)
        assert "server_warnings" not in start_event
        probe = await get_item("LIMIT#startup", "probe")
        assert probe is not None
        expires_at = probe[EXPIRES_AT_ATTRIBUTE]
        assert isinstance(expires_at, int)
        assert expires_at > time()

        async def _denied(*args: Any, **kwargs: Any) -> Item:  # noqa: ANN401, ARG001
            msg = "the role is missing dynamodb:UpdateItem"
            raise TableUnavailableError(msg)

        monkeypatch.setattr(tenant_rate_limits, "add_to_item", _denied)
        await verify_counter_access(start_event)

        assert any(
            "dynamodb:UpdateItem" in str(w) for w in start_event["server_warnings"]
        )

    async def test_the_table_client_gives_up_within_seconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A table that stalls rather than errors still fails closed promptly.

        The counter write sits on the authentication path and every waiter
        of the minute joins it, so the client's own timeout and retries are
        what bound the fail-closed refusal -- the shared default is sized
        for a model answering, minutes long.

        Ref: stdapi/aws.py:AWSConnectionManager.__aenter__
        """
        import stdapi.aws  # noqa: PLC0415
        from stdapi.aws import CONFIG, AWSConnectionManager  # noqa: PLC0415
        from stdapi.config import AWS_SESSION  # noqa: PLC0415

        class _Entered:
            async def __aenter__(self) -> Any:  # noqa: ANN401
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

        recorded: dict[str, Any] = {}

        def _fake_create_client(
            service: str,
            *,
            region_name: str,
            config: Any,  # noqa: ANN401
        ) -> _Entered:
            recorded[f"{service}@{region_name}"] = config
            return _Entered()

        monkeypatch.setattr(stdapi.aws, "_CLIENTS", {})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        monkeypatch.setattr(AWS_SESSION, "create_client", _fake_create_client)
        manager = AWSConnectionManager(("dynamodb", "us-east-1"))

        await manager.__aenter__()
        try:
            config = recorded["dynamodb@us-east-1"]
        finally:
            await manager.__aexit__(None, None, None)

        assert config is not CONFIG
        assert config.read_timeout * config.retries["max_attempts"] <= 30
        assert config.connect_timeout == SETTINGS.aws_connect_timeout


@pytest.mark.gateway("Amazon DynamoDB has no upstream-vendor equivalent")
@pytest.mark.xdist_group("dynamodb")
class TestRealBackends:
    """The atomic grant against the real service, which the stand-in cannot prove.

    Ref: stdapi/tenant_rate_limits.py:_sync
         https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
    """

    async def test_concurrent_reservations_never_exceed_the_limit(
        self, sandbox_dynamodb: str, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Four ledgers racing for one counter never admit more than the limit.

        Every ledger is what one server instance holds; the counter item is
        what they share. Slots are granted in batches, so an instance that
        ran out of requests can strand a few: the sum of admissions is
        bounded by the limit above and only by the stranding below.

        Ref: stdapi/tenant_rate_limits.py:_sync
        """
        from secrets import token_hex  # noqa: PLC0415

        del sandbox_dynamodb
        key_id = token_hex(8)
        tenant = Tenant(key_id=key_id, name="race", requests_per_minute=100)
        ledgers: list[dict[str, Any]] = [{} for _ in range(4)]
        admitted = 0

        async def _instance(ledger: dict[str, Any]) -> int:
            count = 0
            for _ in range(50):
                # Swapped right before the call: the registry is read once,
                # before the first await, so no other instance sees this dict.
                monkeypatch.setattr(tenant_rate_limits, "_LEDGERS", ledger)
                try:
                    await admit_tenant_request(_scope("/v1/chat/completions"), tenant)
                except ApiError as error:
                    # Only the limit refuses: a throttled table is a failure here.
                    if error.status != 429:
                        raise
                    continue
                count += 1
            return count

        try:
            admitted = sum(await gather(*(_instance(ledger) for ledger in ledgers)))
            counter = await _counter(key_id)
        finally:
            for ledger in ledgers:
                for entry in ledger.values():
                    if entry.sync_task is not None:
                        await entry.sync_task

        assert 80 <= admitted <= 100
        assert counter is not None
        requests = counter[REQUESTS_ATTRIBUTE]
        assert isinstance(requests, int)
        assert requests >= admitted
