"""Tests for the per-tenant API keys and their scoped permissions.

The offline lane runs the whole life of a key against local stand-ins: the
operator declares a tenant record, the server mints and delivers the secret,
requests present the key through either header, and the scopes it carries are
enforced at the two choke points -- the ``authenticate`` dependency for
endpoints and ``validate_model`` for models. The adversarial shapes matter
more than the happy path: wrong secrets, revoked keys inside and outside the
cache window, scoped-away models, empty allow lists, malformed keys, and an
unreachable table that must fail closed.

Ref: stdapi/tenant_keys.py
     stdapi/auth.py:authenticate
     https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html
"""

from __future__ import annotations

import re
from asyncio import Event, gather, wait_for
from hmac import compare_digest
from sys import modules
from types import CodeType, ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr, ValidationError

import stdapi.auth
from stdapi import tenant_keys
from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.auth import (
    AuthenticationHandler,
    authenticate,
    verify_websocket_credentials,
)
from stdapi.aws_dynamodb import (
    PARTITION_KEY,
    SORT_KEY,
    TableUnavailableError,
    delete_item,
    get_item,
    put_item,
)
from stdapi.config import SETTINGS, _Settings
from stdapi.monitoring import (
    PRINCIPAL,
    TENANT,
    EventLog,
    Tenant,
    resolve_request_identity,
)
from stdapi.tenant_keys import (
    KEY_PREFIX,
    close_tenant_key_reconciliation,
    initialize_tenant_keys,
    open_tenant_key_reconciliation,
    reconcile_tenant_keys,
    resume_tenant,
    verify_tenant_key,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fastapi.dependencies.models import Dependant
    from types_aiobotocore_ssm.client import SSMClient

    from stdapi.aws_dynamodb import Item, ItemValue

#: A well-formed key ID that no record backs.
_UNKNOWN_KEY_ID = "A" * 16

#: A well-formed secret of the minted length.
_WELL_FORMED_SECRET = "s" * 43

#: Parameter Store prefix the offline lane delivers keys under.
_PREFIX = "/stdapi-test/tenant-keys"


def _request_for(path: str | None) -> Request:
    """Build a request whose matched route reports *path*.

    Args:
        path: The route path template, or None for a routeless request.

    Returns:
        The request.
    """
    scope: dict[str, Any] = {"type": "http"}
    if path is not None:
        scope["route"] = SimpleNamespace(path_format=path)
    return Request(scope)


def _websocket_scope(path: str) -> dict[str, Any]:
    """Build a WebSocket ASGI scope whose matched route reports *path*.

    Args:
        path: The route path template.

    Returns:
        The scope, as the handshake would carry it after routing.
    """
    return {"type": "websocket", "route": SimpleNamespace(path_format=path)}


def _tenant_item(key_id: str, **attributes: ItemValue) -> Item:
    """Build the tenant record the Terraform module would write.

    Args:
        key_id: The tenant's key ID.
        **attributes: Extra record attributes, e.g. scope lists.

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
async def tenant_backend(
    dynamodb_table: str,  # noqa: ARG001 - binds the table stand-in
    moto_dynamodb_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[SSMClient]:
    """Enable tenant keys against the local table and Parameter Store stand-ins.

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
        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant_keys._NEGATIVE.clear()  # noqa: SLF001
        tenant_keys._REPORTED.clear()  # noqa: SLF001
        token = TENANT.set(None)
        try:
            yield ssm_client
        finally:
            TENANT.reset(token)
            tenant_keys._CACHE.clear()  # noqa: SLF001
            tenant_keys._NEGATIVE.clear()  # noqa: SLF001
            tenant_keys._REPORTED.clear()  # noqa: SLF001


async def _declare_and_mint(
    ssm_client: SSMClient, key_id: str = "k" + "0" * 15, **attributes: ItemValue
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


class TestDeliveryKeyConfiguration:
    """What the settings model accepts for the delivery encryption key.

    Ref: stdapi/config.py:_validate_tenant_keys
         https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_PutParameter.html
    """

    @pytest.mark.parametrize(
        "key_id",
        [
            "12345678-1234-1234-1234-123456789012",
            "mrk-1234567890abcdef1234567890abcdef",
            "alias/stdapi-ai",
            (
                "arn:aws:kms:us-east-1:123456789012:key/"
                "12345678-1234-1234-1234-123456789012"
            ),
            "arn:aws-us-gov:kms:us-gov-west-1:123456789012:alias/stdapi-ai",
            "alias/prod:tenant-keys",
        ],
    )
    def test_every_form_of_a_key_reference_is_accepted(self, key_id: str) -> None:
        """AWS takes a key id, an alias or an ARN of either, so this does too."""
        settings = _Settings(
            tenant_api_keys=True,
            aws_dynamodb_table="shared",
            tenant_key_ssm_parameter_prefix="/test/tenant-keys",
            tenant_key_ssm_kms_key_id=key_id,
        )

        assert settings.tenant_key_ssm_kms_key_id == key_id

    def test_a_malformed_key_reference_fails_startup(self) -> None:
        """A typo would otherwise only surface when the first tenant is minted."""
        with pytest.raises(ValidationError, match="tenant_key_ssm_kms_key_id"):
            _Settings(
                tenant_api_keys=True,
                aws_dynamodb_table="shared",
                tenant_key_ssm_parameter_prefix="/test/tenant-keys",
                tenant_key_ssm_kms_key_id="not a key",
            )

    def test_a_key_reference_past_the_service_limit_fails_startup(self) -> None:
        """Parameter Store caps KeyId at 256 characters, which the shape allows past.

        Ref: https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_PutParameter.html
        """
        with pytest.raises(ValidationError, match="tenant_key_ssm_kms_key_id"):
            _Settings(
                tenant_api_keys=True,
                aws_dynamodb_table="shared",
                tenant_key_ssm_parameter_prefix="/test/tenant-keys",
                tenant_key_ssm_kms_key_id=(
                    "arn:aws:kms:us-east-1:123456789012:alias/" + "a" * 250
                ),
            )

    def test_the_key_without_the_feature_fails_startup(self) -> None:
        """Nothing is ever delivered under it, so the setting is a mistake."""
        with pytest.raises(ValidationError, match="requires tenant_api_keys"):
            _Settings(tenant_key_ssm_kms_key_id="alias/stdapi-ai")


def _capture_put_parameter(
    ssm_client: SSMClient, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Record the delivery requests the server sends, and send them on.

    Args:
        ssm_client: The Parameter Store stand-in client.
        monkeypatch: The patcher restoring the client afterwards.

    Returns:
        The keyword arguments of each ``PutParameter`` call, in order.
    """
    requests: list[dict[str, Any]] = []
    original = ssm_client.put_parameter

    async def _record(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
        requests.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(ssm_client, "put_parameter", _record)
    return requests


class TestScopePatterns:
    """The allow/deny semantics of one tenant's scope lists.

    Ref: stdapi/monitoring.py:Tenant
    """

    def test_no_lists_restrict_nothing(self) -> None:
        """A tenant declared without scopes may use any model and endpoint."""
        tenant = Tenant(key_id=_UNKNOWN_KEY_ID, name="t")

        assert tenant.allows_model("anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert tenant.allows_endpoint("/v1/chat/completions")

    def test_an_empty_allow_list_allows_nothing(self) -> None:
        """Explicitly declaring no allowed models grants no model at all."""
        tenant = Tenant(key_id=_UNKNOWN_KEY_ID, name="t", models_allow=())

        assert not tenant.allows_model("amazon.nova-lite-v1:0")
        assert not tenant.allows_model("")

    def test_a_deny_wins_over_a_matching_allow(self) -> None:
        """The deny list is checked first, so it cannot be reopened by allow."""
        tenant = Tenant(
            key_id=_UNKNOWN_KEY_ID,
            name="t",
            models_allow=("anthropic.*",),
            models_deny=("anthropic.claude-opus*",),
        )

        assert tenant.allows_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        assert not tenant.allows_model("anthropic.claude-opus-4-6-v1:0")

    def test_endpoint_globs_cover_path_parameters(self) -> None:
        """A ``*`` pattern matches the templated part of a route path."""
        tenant = Tenant(
            key_id=_UNKNOWN_KEY_ID,
            name="t",
            endpoints_allow=("/v1/files/*", "/v1/models"),
        )

        assert tenant.allows_endpoint("/v1/files/{file_id}")
        assert tenant.allows_endpoint("/v1/models")
        assert not tenant.allows_endpoint("/v1/chat/completions")

    def test_a_deny_only_endpoint_list_restricts_that_route_alone(self) -> None:
        """Without an allow list, a deny list still closes what it names."""
        tenant = Tenant(
            key_id=_UNKNOWN_KEY_ID, name="t", endpoints_deny=("/v1/files/*",)
        )

        assert not tenant.allows_endpoint("/v1/files/{file_id}")
        assert tenant.allows_endpoint("/v1/chat/completions")


class TestMalformedKeys:
    """A key that is not the minted shape is refused before any lookup.

    Ref: stdapi/tenant_keys.py:_parse
    """

    @pytest.mark.parametrize(
        "credential",
        [
            "sk-std-",
            "sk-std-short-secret",
            f"sk-std-{'A' * 16}x{'s' * 43}",
            f"sk-std-{'A' * 16}-{'s' * 42}",
            f"sk-std-{'A' * 15}é-{'s' * 43}",
            f"sk-std-{'A' * 16}-{'s' * 42}.",
            f"sk-std-{'A' * 16}-{'s' * 100}",
        ],
    )
    async def test_a_malformed_key_is_a_plain_401(self, credential: str) -> None:
        """No shape of garbage reaches the table or leaks why it was refused."""
        with pytest.raises(ApiError) as raised:
            await verify_tenant_key(credential)

        assert raised.value.status == 401
        assert str(raised.value) == "Unauthorized"


class TestMinting:
    """A declared tenant gets its secret minted, delivered once, and recorded.

    Ref: stdapi/tenant_keys.py:reconcile_tenant_keys
         https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html
    """

    async def test_a_pending_tenant_is_minted_and_its_key_validates(
        self, tenant_backend: SSMClient
    ) -> None:
        """The delivered parameter holds a key the gateway then accepts."""
        key = await _declare_and_mint(tenant_backend)

        tenant = await verify_tenant_key(key)

        assert tenant.key_id == "k" + "0" * 15
        assert tenant.name == "tenant-k" + "0" * 15

    async def test_the_stored_record_holds_a_hash_and_never_the_secret(
        self, tenant_backend: SSMClient
    ) -> None:
        """The table carries a salted digest; the key itself appears nowhere."""
        key = await _declare_and_mint(tenant_backend, key_id="h" + "0" * 15)

        item = await get_item("TENANT", "secret#h" + "0" * 15)

        assert item is not None
        assert isinstance(item["secret_hash"], bytes)
        assert isinstance(item["salt"], bytes)
        secret = key.rsplit("-", 1)[-1]
        assert secret not in str(item)

    async def test_minting_is_idempotent_across_runs(
        self, tenant_backend: SSMClient
    ) -> None:
        """A second reconciliation changes neither the parameter nor the hash."""
        key = await _declare_and_mint(tenant_backend, key_id="i" + "0" * 15)

        await reconcile_tenant_keys()

        response = await tenant_backend.get_parameter(
            Name=f"{_PREFIX}/i{'0' * 15}", WithDecryption=True
        )
        assert response["Parameter"]["Value"] == key
        assert response["Parameter"]["Version"] == 1

    async def test_a_crash_between_delivery_and_recording_recovers(
        self, tenant_backend: SSMClient
    ) -> None:
        """The parameter is the source of truth: the hash is rebuilt from it."""
        key = await _declare_and_mint(tenant_backend, key_id="c" + "0" * 15)
        await delete_item("TENANT", "secret#c" + "0" * 15)

        await reconcile_tenant_keys()

        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant_keys._NEGATIVE.clear()  # noqa: SLF001
        assert (await verify_tenant_key(key)).key_id == "c" + "0" * 15

    async def test_a_destroyed_tenant_loses_its_credential_record(
        self, tenant_backend: SSMClient
    ) -> None:
        """Reconciliation drops the orphaned hash once the tenant is gone."""
        key = await _declare_and_mint(tenant_backend, key_id="d" + "0" * 15)
        await delete_item("TENANT", "tenant#d" + "0" * 15)

        await reconcile_tenant_keys()

        assert await get_item("TENANT", "secret#d" + "0" * 15) is None
        tenant_keys._CACHE.clear()  # noqa: SLF001
        with pytest.raises(ApiError):
            await verify_tenant_key(key)

    async def test_a_parameter_holding_something_else_is_never_recorded(
        self, tenant_backend: SSMClient
    ) -> None:
        """An operator-planted value cannot become a tenant's credential."""
        await tenant_backend.put_parameter(
            Name=f"{_PREFIX}/j{'0' * 15}", Value="not-a-key", Type="SecureString"
        )
        await put_item(_tenant_item("j" + "0" * 15))

        await reconcile_tenant_keys()

        assert await get_item("TENANT", "secret#j" + "0" * 15) is None

    async def test_the_configured_kms_key_encrypts_the_delivery_parameter(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The setting reaches Parameter Store as the parameter's KeyId.

        Encrypting the delivery under the deployment's own key is what makes
        reading it back require kms:Decrypt on that key, instead of the
        account-wide reach of the AWS-managed 'alias/aws/ssm' key.

        Ref: https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_PutParameter.html
             https://docs.aws.amazon.com/systems-manager/latest/userguide/secure-string-parameter-kms-encryption.html
        """
        monkeypatch.setattr(SETTINGS, "tenant_key_ssm_kms_key_id", "alias/stdapi-ai")
        requests = _capture_put_parameter(tenant_backend, monkeypatch)
        await put_item(_tenant_item("m" + "0" * 15))

        await reconcile_tenant_keys()

        assert requests[0]["KeyId"] == "alias/stdapi-ai"

    async def test_no_configured_kms_key_leaves_the_field_out(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset sends no KeyId at all, which is what selects 'alias/aws/ssm'.

        Sending KeyId=None instead would be rejected by the parameter
        validation of the request, not treated as absent.

        Ref: https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_PutParameter.html
        """
        requests = _capture_put_parameter(tenant_backend, monkeypatch)
        await put_item(_tenant_item("n" + "0" * 15))

        await reconcile_tenant_keys()

        assert SETTINGS.tenant_key_ssm_kms_key_id is None
        assert "KeyId" not in requests[0]


class TestVerification:
    """The adversarial shapes a validator must refuse, and how fast it answers.

    Ref: stdapi/tenant_keys.py:verify_tenant_key
    """

    async def test_a_wrong_secret_is_refused_in_constant_time(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong secret of the right shape is a 401, decided by compare_digest."""
        key = await _declare_and_mint(tenant_backend, key_id="w" + "0" * 15)
        compared: list[tuple[bytes, bytes]] = []

        def _spy(left: bytes, right: bytes) -> bool:
            compared.append((left, right))
            return compare_digest(left, right)

        monkeypatch.setattr(tenant_keys, "compare_digest", _spy)
        wrong = key[:-4] + ("AAAA" if not key.endswith("AAAA") else "BBBB")

        with pytest.raises(ApiError) as raised:
            await verify_tenant_key(wrong)

        assert raised.value.status == 401
        assert str(raised.value) == "Unauthorized"
        assert compared, "the secret must be compared with hmac.compare_digest"

    async def test_an_unknown_key_id_is_refused_and_negative_cached(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second flood of the same fabricated key never reaches the table."""
        del tenant_backend
        reads: list[str] = []

        async def _spy(
            partition_key: str, sort_key: str, *, consistent: bool = False
        ) -> Item | None:
            reads.append(sort_key)
            return await get_item(partition_key, sort_key, consistent=consistent)

        monkeypatch.setattr(tenant_keys, "get_item", _spy)
        credential = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"

        with pytest.raises(ApiError):
            await verify_tenant_key(credential)
        first = len(reads)
        with pytest.raises(ApiError):
            await verify_tenant_key(credential)

        assert first == 2
        assert len(reads) == first

    async def test_the_negative_cache_is_bounded(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flood of distinct fabricated keys cannot grow memory unbounded."""
        del tenant_backend
        monkeypatch.setattr(tenant_keys, "_NEGATIVE_MAX", 8)
        for index in range(20):
            with pytest.raises(ApiError):
                await verify_tenant_key(
                    f"{KEY_PREFIX}{index:016d}-{_WELL_FORMED_SECRET}"
                )

        assert len(tenant_keys._NEGATIVE) <= 8  # noqa: SLF001

    async def test_an_unknown_key_id_costs_the_same_work_as_a_wrong_secret(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fabricated key ID is hashed and compared like a real one.

        Refusing it without that work would time-leak which key IDs exist,
        which is the property the module docstring claims to close -- on the
        negative-cached refusal as much as on the cold one.

        Ref: stdapi/tenant_keys.py:_reject_unknown
        """
        del tenant_backend
        compared: list[tuple[bytes, bytes]] = []

        def _spy(left: bytes, right: bytes) -> bool:
            compared.append((left, right))
            return compare_digest(left, right)

        monkeypatch.setattr(tenant_keys, "compare_digest", _spy)
        credential = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"

        with pytest.raises(ApiError):
            await verify_tenant_key(credential)
        cold = len(compared)
        with pytest.raises(ApiError):
            await verify_tenant_key(credential)

        assert cold == 1, "the cold refusal must hash and compare the secret"
        assert len(compared) == 2, "the cached refusal must do the same work"
        assert all(len(left) == len(right) for left, right in compared)

    async def test_the_negative_cache_expires(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key refused before it was minted is retried, not shut out for long.

        Ref: stdapi/tenant_keys.py:_NEGATIVE_TTL
        """
        del tenant_backend
        reads: list[str] = []

        async def _spy(
            partition_key: str, sort_key: str, *, consistent: bool = False
        ) -> Item | None:
            reads.append(sort_key)
            return await get_item(partition_key, sort_key, consistent=consistent)

        monkeypatch.setattr(tenant_keys, "get_item", _spy)
        monkeypatch.setattr(tenant_keys, "_NEGATIVE_TTL", 0.0)
        credential = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"

        with pytest.raises(ApiError):
            await verify_tenant_key(credential)
        with pytest.raises(ApiError):
            await verify_tenant_key(credential)

        assert len(reads) == 4, "an expired negative entry must re-read the table"

    async def test_the_positive_cache_is_bounded(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verified keys cannot accumulate without bound in a long-lived process.

        Ref: stdapi/tenant_keys.py:_CACHE_MAX
        """
        monkeypatch.setattr(tenant_keys, "_CACHE_MAX", 2)
        for index in range(4):
            key = await _declare_and_mint(tenant_backend, key_id=f"t{index}{'0' * 14}")
            await verify_tenant_key(key)

        assert len(tenant_keys._CACHE) <= 2  # noqa: SLF001

    async def test_a_disabled_key_is_refused(self, tenant_backend: SSMClient) -> None:
        """The operator's ``disabled`` flag turns the key off without deleting it."""
        key = await _declare_and_mint(
            tenant_backend, key_id="x" + "0" * 15, disabled=True
        )

        with pytest.raises(ApiError) as raised:
            await verify_tenant_key(key)

        assert raised.value.status == 401

    async def test_a_revoked_key_keeps_working_inside_the_cache_window(
        self, tenant_backend: SSMClient
    ) -> None:
        """Within ``tenant_key_cache_seconds`` the last decision stands, by design."""
        key = await _declare_and_mint(tenant_backend, key_id="r" + "0" * 15)
        await verify_tenant_key(key)
        await delete_item("TENANT", "tenant#r" + "0" * 15)
        await delete_item("TENANT", "secret#r" + "0" * 15)

        assert (await verify_tenant_key(key)).key_id == "r" + "0" * 15

    async def test_a_revoked_key_is_refused_outside_the_cache_window(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the cache expires, the revocation is final on every instance."""
        key = await _declare_and_mint(tenant_backend, key_id="e" + "0" * 15)
        await verify_tenant_key(key)
        await delete_item("TENANT", "tenant#e" + "0" * 15)
        monkeypatch.setattr(SETTINGS, "tenant_key_cache_seconds", 0.0)

        with pytest.raises(ApiError) as raised:
            await verify_tenant_key(key)

        assert raised.value.status == 401

    async def test_an_unreachable_table_fails_closed_as_a_503(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid-shaped key is neither accepted nor called wrong: it is a 503."""
        del tenant_backend

        async def _unreachable(*_args: object, **_kwargs: object) -> Item | None:
            detail = "the table is unreachable"
            raise TableUnavailableError(detail)

        monkeypatch.setattr(tenant_keys, "get_item", _unreachable)

        with pytest.raises(FeatureUnavailableError):
            await verify_tenant_key(
                f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"
            )

    async def test_a_record_from_a_newer_build_is_refused_as_unavailable(
        self, tenant_backend: SSMClient
    ) -> None:
        """A rolling deployment's newer schema is an operator matter, not a 401."""
        key_id = "n" + "0" * 15
        key = await _declare_and_mint(tenant_backend, key_id=key_id)
        await put_item(_tenant_item(key_id, schema=99))
        tenant_keys._CACHE.clear()  # noqa: SLF001

        with pytest.raises(FeatureUnavailableError):
            await verify_tenant_key(key)

    async def test_a_malformed_scope_attribute_is_refused_as_unavailable(
        self, tenant_backend: SSMClient
    ) -> None:
        """A record this build cannot interpret never resolves to an open scope."""
        key_id = "m" + "0" * 15
        key = await _declare_and_mint(tenant_backend, key_id=key_id)
        await put_item(_tenant_item(key_id, models_allow="not-a-list"))
        tenant_keys._CACHE.clear()  # noqa: SLF001

        with pytest.raises(FeatureUnavailableError):
            await verify_tenant_key(key)

    async def test_a_resumed_grant_follows_the_tenant_lifecycle(
        self, tenant_backend: SSMClient
    ) -> None:
        """A minted Realtime secret stops resuming once the key is disabled or gone.

        Ref: stdapi/tenant_keys.py:resume_tenant
             stdapi/realtime.py:read_client_secret
        """
        key_id = "g" + "0" * 15
        await _declare_and_mint(tenant_backend, key_id=key_id)

        assert (await resume_tenant(key_id)).key_id == key_id

        await put_item(_tenant_item(key_id, disabled=True))
        tenant_keys._CACHE.clear()  # noqa: SLF001
        with pytest.raises(ApiError):
            await resume_tenant(key_id)

        await delete_item("TENANT", "tenant#" + key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        with pytest.raises(ApiError):
            await resume_tenant(key_id)

    async def test_a_resumed_grant_refuses_a_malformed_key_id(self) -> None:
        """A key ID that is not the minted shape never reaches the table.

        Ref: stdapi/tenant_keys.py:resume_tenant
        """
        with pytest.raises(ApiError) as raised:
            await resume_tenant("too-short")

        assert raised.value.status == 401


class TestAuthenticateDispatch:
    """How tenant keys ride the two headers next to the other credential kinds.

    Ref: stdapi/auth.py:authenticate
    """

    @staticmethod
    def _static_handler(monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
        """Install a deployment key handler accepting only *api_key*."""
        handler = AuthenticationHandler()
        handler._hash_api_key(SecretStr(api_key))  # noqa: SLF001
        monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)

    async def test_a_tenant_key_works_in_either_header(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI SDKs send ``Authorization: Bearer``; both carriers verify."""
        self._static_handler(monkeypatch, "good-key")
        key = await _declare_and_mint(tenant_backend, key_id="a" + "0" * 15)

        await authenticate(credentials=None, x_api_key=key)
        assert TENANT.get() is not None
        await authenticate(
            credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=key),
            x_api_key=None,
        )
        assert TENANT.get() is not None

    async def test_a_tenant_key_and_a_deployment_key_both_verify(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tenant key in x-api-key no longer silences the bearer credential."""
        self._static_handler(monkeypatch, "good-key")
        key = await _declare_and_mint(tenant_backend, key_id="b" + "0" * 15)

        await authenticate(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="good-key"
            ),
            x_api_key=key,
        )

        assert TENANT.get() is not None

    async def test_a_tenant_key_with_a_wrong_bearer_is_refused(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both presented credentials must hold: a bad companion fails the request."""
        self._static_handler(monkeypatch, "good-key")
        key = await _declare_and_mint(tenant_backend, key_id="f" + "0" * 15)

        with pytest.raises(ApiError):
            await authenticate(
                credentials=HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials="wrong-key"
                ),
                x_api_key=key,
            )

    async def test_the_x_api_key_tenant_wins_over_a_bearer_tenant(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two tenant keys, two headers: the request is scoped to ``x-api-key``.

        The bearer still has to verify, but it only identifies. Letting it
        re-scope the request would attribute and bill it to the wrong tenant --
        the failure a proxy injecting its own ``x-api-key`` would produce.
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        first = await _declare_and_mint(tenant_backend, key_id="1" + "0" * 15)
        second = await _declare_and_mint(tenant_backend, key_id="2" + "0" * 15)

        await authenticate(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=second
            ),
            x_api_key=first,
        )

        tenant = TENANT.get()
        assert tenant is not None
        assert tenant.key_id == "1" + "0" * 15

    async def test_todays_x_api_key_precedence_is_untouched(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment key in x-api-key still wins over any bearer, as before."""
        del tenant_backend
        self._static_handler(monkeypatch, "good-key")

        await authenticate(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="wrong-key"
            ),
            x_api_key="good-key",
        )

        assert TENANT.get() is None

    async def test_a_tenant_shaped_deployment_key_keeps_working(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A static key that happens to look like a tenant key is not broken."""
        del tenant_backend
        lookalike = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"
        self._static_handler(monkeypatch, lookalike)

        await authenticate(credentials=None, x_api_key=lookalike)

        assert TENANT.get() is None

    async def test_tenant_only_auth_never_falls_open(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no deployment key, a non-tenant credential is refused, not ignored."""
        del tenant_backend
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())

        with pytest.raises(ApiError):
            await authenticate(credentials=None, x_api_key="anything")
        with pytest.raises(ApiError):
            await authenticate(credentials=None, x_api_key=None)


class TestEndpointScope:
    """Endpoint restrictions are enforced inside the dependency itself.

    Ref: stdapi/auth.py:enforce_tenant_endpoint_scope
    """

    async def test_an_out_of_scope_endpoint_is_refused(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allow list decides per matched route template."""
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        key = await _declare_and_mint(
            tenant_backend,
            key_id="p" + "0" * 15,
            endpoints_allow=["/v1/chat/completions"],
        )

        await authenticate(
            credentials=None,
            x_api_key=key,
            request=_request_for("/v1/chat/completions"),
        )
        with pytest.raises(ApiError) as raised:
            await authenticate(
                credentials=None, x_api_key=key, request=_request_for("/v1/embeddings")
            )

        assert raised.value.status == 401

    async def test_a_restricted_tenant_fails_closed_without_a_route(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No matched route to test means no access, never a skipped check."""
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        key = await _declare_and_mint(
            tenant_backend,
            key_id="q" + "0" * 15,
            endpoints_allow=["/v1/chat/completions"],
        )

        with pytest.raises(ApiError):
            await authenticate(
                credentials=None, x_api_key=key, request=_request_for(None)
            )

    async def test_a_deny_only_tenant_is_refused_on_the_denied_route(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tenant declaring only ``endpoints_deny`` is still restricted."""
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        key = await _declare_and_mint(
            tenant_backend, key_id="z" + "0" * 15, endpoints_deny=["/v1/files/*"]
        )

        await authenticate(
            credentials=None,
            x_api_key=key,
            request=_request_for("/v1/chat/completions"),
        )
        with pytest.raises(ApiError) as raised:
            await authenticate(
                credentials=None,
                x_api_key=key,
                request=_request_for("/v1/files/{file_id}"),
            )

        assert raised.value.status == 401

    async def test_a_websocket_outside_the_scope_is_refused(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scope of the matched WebSocket route is enforced at the handshake.

        Ref: stdapi/auth.py:verify_websocket_credentials
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        key = await _declare_and_mint(
            tenant_backend,
            key_id="w" + "1" * 15,
            endpoints_allow=["/v1/chat/completions"],
        )

        await verify_websocket_credentials(
            key, _websocket_scope("/v1/chat/completions")
        )
        with pytest.raises(ApiError) as raised:
            await verify_websocket_credentials(key, _websocket_scope("/v1/realtime"))

        assert raised.value.status == 401

    async def test_a_websocket_without_a_matched_route_fails_closed(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No scope means no route to test, which must refuse a restricted tenant.

        Ref: stdapi/auth.py:verify_websocket_credentials
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        key = await _declare_and_mint(
            tenant_backend,
            key_id="w" + "2" * 15,
            endpoints_allow=["/v1/chat/completions"],
        )

        with pytest.raises(ApiError) as raised:
            await verify_websocket_credentials(key)

        assert raised.value.status == 401

    async def test_the_matched_route_reaches_the_dependency_through_the_app(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real ASGI app fills the dependency's request with the matched route.

        Every other scope test calls ``authenticate`` directly with a request
        built by hand; this one proves FastAPI still fills that parameter, so a
        framework change that stopped doing it cannot silently refuse every
        scoped tenant on every route.

        Ref: stdapi/auth.py:authenticate
        """
        from httpx import ASGITransport, AsyncClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        key = await _declare_and_mint(
            tenant_backend,
            key_id="u" + "0" * 15,
            endpoints_allow=["/v1/chat/completions"],
        )
        headers = {"x-api-key": key}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            denied = await client.post("/v1/embeddings", json={}, headers=headers)
            allowed = await client.post(
                "/v1/chat/completions", json={}, headers=headers
            )

        assert denied.status_code == 401
        # Refused by the body schema, which only runs once the scope allowed it.
        assert allowed.status_code == 400


class TestModelScope:
    """Model restrictions are enforced at the single model-resolution point.

    Ref: stdapi/models/__init__.py:validate_model
    """

    @staticmethod
    def _catalog(
        monkeypatch: pytest.MonkeyPatch, model_id: str, output: str = "TEXT"
    ) -> None:
        """Install a one-model catalog so validation runs offline."""
        import stdapi.models as models_module  # noqa: PLC0415
        from stdapi.models import ModelDetails  # noqa: PLC0415

        async def _no_refresh(_start_event: object = None) -> bool:
            return False

        details = ModelDetails(
            id=model_id,
            name="Test model",
            provider="Test",
            input_modalities=["TEXT"],
            output_modalities=[output],
            regions=[SETTINGS.aws_bedrock_regions[0]],
        )
        monkeypatch.setattr(models_module, "_MODELS", {model_id: details})
        monkeypatch.setattr(models_module, "_refresh_bedrock_models", _no_refresh)

    async def test_a_scoped_away_model_is_the_not_found_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """To a tenant, a denied model is indistinguishable from a missing one."""
        import stdapi.models as models_module  # noqa: PLC0415
        from stdapi.api_errors import UnsupportedModelError  # noqa: PLC0415
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        self._catalog(monkeypatch, "provider.denied-model-v1:0")
        token = REQUEST_LOG.set({"type": "request", "level": "info"})  # type: ignore[typeddict-item]
        tenant_token = TENANT.set(
            Tenant(key_id=_UNKNOWN_KEY_ID, name="t", models_deny=("provider.*",))
        )
        try:
            with pytest.raises(UnsupportedModelError) as raised:
                await models_module.validate_model("provider.denied-model-v1:0")
            assert raised.value.status == 404

            TENANT.set(Tenant(key_id=_UNKNOWN_KEY_ID, name="t"))
            model = await models_module.validate_model("provider.denied-model-v1:0")
            assert model.id == "provider.denied-model-v1:0"
        finally:
            TENANT.reset(tenant_token)
            REQUEST_LOG.reset(token)

    async def test_an_alias_cannot_launder_a_denied_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scope is checked on the resolved ID, after alias resolution."""
        import stdapi.models as models_module  # noqa: PLC0415
        from stdapi.api_errors import UnsupportedModelError  # noqa: PLC0415
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        self._catalog(monkeypatch, "provider.denied-model-v1:0")
        monkeypatch.setitem(
            models_module.MODEL_ALIASES, "friendly-alias", "provider.denied-model-v1:0"
        )
        token = REQUEST_LOG.set({"type": "request", "level": "info"})  # type: ignore[typeddict-item]
        tenant_token = TENANT.set(
            Tenant(
                key_id=_UNKNOWN_KEY_ID,
                name="t",
                models_deny=("provider.denied-model-v1:0",),
            )
        )
        try:
            with pytest.raises(UnsupportedModelError):
                await models_module.validate_model("friendly-alias")
        finally:
            TENANT.reset(tenant_token)
            REQUEST_LOG.reset(token)

    async def test_the_moderations_route_applies_the_model_scope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/v1/moderations` refuses a moderation model outside the scope.

        The route picks its backend itself instead of resolving one through
        `validate_model`, so it is the one place the documented choke point
        does not run for it: without a check of its own, a tenant scoped to
        named models still reaches Amazon Comprehend and Bedrock Guardrails on
        the deployment's bill.

        Ref: https://stdapi.ai/operations_authentication_security/#scopes
             stdapi/routes/openai_moderations.py:create_moderation
        """
        from stdapi.api_errors import UnsupportedModelError  # noqa: PLC0415
        from stdapi.aws_bedrock import COMPREHEND_MODERATION_MODEL  # noqa: PLC0415
        from stdapi.monitoring import REQUEST_ID, REQUEST_LOG  # noqa: PLC0415
        from stdapi.routes.openai_moderations import create_moderation  # noqa: PLC0415
        from stdapi.types.openai_moderations import (  # noqa: PLC0415
            Moderation,
            ModerationCategories,
            ModerationCategoryScores,
            ModerationCreateParams,
        )

        async def _moderate(_self: object, _item: object) -> Moderation:
            return Moderation(
                flagged=False,
                categories=ModerationCategories(),
                category_scores=ModerationCategoryScores(),
            )

        monkeypatch.setattr(
            "stdapi.models.moderation.amazon_comprehend.ModerationModel.moderate",
            _moderate,
        )
        body = ModerationCreateParams(
            input="some text", model=COMPREHEND_MODERATION_MODEL
        )
        token = REQUEST_LOG.set({"type": "request", "level": "info"})  # type: ignore[typeddict-item]
        request_token = REQUEST_ID.set("test-request")
        tenant_token = TENANT.set(
            Tenant(key_id=_UNKNOWN_KEY_ID, name="t", models_allow=("anthropic.*",))
        )
        try:
            with pytest.raises(UnsupportedModelError) as raised:
                await create_moderation(body)
            assert raised.value.status == 404

            TENANT.set(
                Tenant(
                    key_id=_UNKNOWN_KEY_ID,
                    name="t",
                    models_allow=(COMPREHEND_MODERATION_MODEL,),
                )
            )
            answer = await create_moderation(body)
            assert answer.model == COMPREHEND_MODERATION_MODEL
        finally:
            TENANT.reset(tenant_token)
            REQUEST_ID.reset(request_token)
            REQUEST_LOG.reset(token)

    async def test_the_deployments_embedding_model_is_outside_the_scope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creating a vector store does not need the operator's model in scope.

        The embedding model of a vector store is the deployment's setting, one
        the request never names and cannot choose; searching or indexing an
        existing store resolves it unchecked. Applying the scope on creation
        alone would answer a tenant `404 model_not_found` for an identifier it
        never sent, for a store it can still search.

        Ref: https://stdapi.ai/operations_authentication_security/#scopes
             stdapi/vector_stores/engine.py:resolve_embedding_model
        """
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415
        from stdapi.vector_stores import engine  # noqa: PLC0415

        model_id = SETTINGS.vector_store_embedding_model
        self._catalog(monkeypatch, model_id, output="EMBEDDING")

        async def _embed(_model_id: str, texts: list[str]) -> list[list[float]]:
            return [[0.0, 1.0] for _ in texts]

        monkeypatch.setattr(engine, "_embed", _embed)
        token = REQUEST_LOG.set({"type": "request", "level": "info"})  # type: ignore[typeddict-item]
        tenant_token = TENANT.set(
            Tenant(key_id=_UNKNOWN_KEY_ID, name="t", models_allow=("anthropic.*",))
        )
        try:
            assert await engine.resolve_embedding_model() == (model_id, 2)
        finally:
            TENANT.reset(tenant_token)
            REQUEST_LOG.reset(token)


class TestIdentityAndBinding:
    """How a verified tenant reaches attribution and minted Realtime secrets.

    Ref: stdapi/monitoring.py:resolve_request_identity
         stdapi/realtime.py:mint_client_secret
    """

    def test_the_tenant_is_the_identity_of_last_resort(self) -> None:
        """Principal, then the declared user, then the tenant key ID."""
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        log_token = REQUEST_LOG.set({"type": "request", "level": "info"})  # type: ignore[typeddict-item]
        principal_token = PRINCIPAL.set(None)
        tenant_token = TENANT.set(Tenant(key_id=_UNKNOWN_KEY_ID, name="t"))
        try:
            assert resolve_request_identity() == _UNKNOWN_KEY_ID
            REQUEST_LOG.get()["request_user_id"] = "declared-user"
            assert resolve_request_identity() == "declared-user"
        finally:
            TENANT.reset(tenant_token)
            PRINCIPAL.reset(principal_token)
            REQUEST_LOG.reset(log_token)

    def test_a_minted_realtime_secret_is_bound_to_its_tenant(self) -> None:
        """The signed secret carries the key ID, so the session keeps the scopes."""
        from stdapi.realtime import (  # noqa: PLC0415
            mint_client_secret,
            read_client_secret,
        )
        from stdapi.types.openai_realtime import RealtimeSessionConfig  # noqa: PLC0415

        tenant_token = TENANT.set(Tenant(key_id=_UNKNOWN_KEY_ID, name="t"))
        try:
            value, _ = mint_client_secret(RealtimeSessionConfig(), 600)
        finally:
            TENANT.reset(tenant_token)

        read = read_client_secret(value)
        assert read is not None
        assert read.tenant_key_id == _UNKNOWN_KEY_ID


def _reaches_websocket_verification(endpoint: Callable[..., Any]) -> bool:
    """Whether *endpoint* can reach :func:`verify_websocket_credentials`.

    Walks the call graph statically, following only the names this package's
    own modules bind, so a handler delegating its handshake to a helper still
    counts.

    Args:
        endpoint: The WebSocket route's handler.

    Returns:
        True when the verification is reachable from the handler.
    """
    seen: set[CodeType] = set()

    def _walk(code: CodeType, module: ModuleType | None) -> bool:
        if code in seen:
            return False
        seen.add(code)
        for name in code.co_names:
            target = getattr(module, name, None)
            if target is verify_websocket_credentials:
                return True
            called = getattr(target, "__code__", None)
            if (
                called is not None
                and getattr(target, "__module__", "").startswith("stdapi")
                and _walk(called, modules.get(target.__module__))
            ):
                return True
        return any(
            _walk(nested, module)
            for nested in code.co_consts
            if isinstance(nested, CodeType)
        )

    return _walk(endpoint.__code__, modules.get(endpoint.__module__))


class TestEveryRouteIsAuthenticated:
    """The guarantee that a new route cannot silently bypass authentication.

    Ref: stdapi/auth.py:authenticate
         stdapi/routes/__init__.py:discover_routers
    """

    #: Route paths that deliberately answer without a credential.
    _ALLOWLIST = re.compile(
        r"^("
        r"/health|/ping"  # liveness probes, must answer before auth is up
        r"|/favicon\.ico|/robots\.txt|/docs-assets/\{name\}"  # static assets
        r"|/"  # the landing page, which carries no data
        r"|/\.well-known/.*"  # discovery metadata, public by protocol
        r")$"
    )

    #: WebSocket routes, which authenticate by hand rather than by dependency.
    _WEBSOCKET_ROUTES = frozenset({"/v1/realtime"})

    def test_every_route_carries_the_authenticate_dependency(self) -> None:
        """Endpoint scopes are only real if no route can forget to authenticate."""
        from fastapi.routing import APIRoute  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        def _has_authenticate(dependant: Dependant) -> bool:
            return any(
                sub.call is authenticate or _has_authenticate(sub)
                for sub in dependant.dependencies
            )

        # Routers are included lazily: each entry expands to the HTTP routes it
        # serves, with the prefixed path template the dependency will also see.
        checked: list[tuple[str, Dependant]] = []
        for route in app.routes:
            if isinstance(route, APIRoute):
                checked.append((route.path_format, route.dependant))
            elif hasattr(route, "effective_route_contexts"):
                checked.extend(
                    (context.path_format, context.dependant)
                    for context in route.effective_route_contexts()
                    if context.methods
                )
        unprotected = sorted(
            {
                path
                for path, dependant in checked
                if not _has_authenticate(dependant) and not self._ALLOWLIST.match(path)
            }
        )

        assert len(checked) > 50, "the app under test must carry its routes"
        assert unprotected == []

    def test_every_websocket_route_verifies_its_credential(self) -> None:
        """A WebSocket route cannot be added without a credential check either.

        ``Depends(authenticate)`` never runs on a WebSocket handshake, so these
        routes call :func:`verify_websocket_credentials` themselves -- exactly
        the kind of call a new route forgets. Both halves are asserted: the set
        of WebSocket paths is declared here, so a new one fails by default, and
        each declared handler must still reach the verification.

        Ref: stdapi/auth.py:verify_websocket_credentials
        """
        from fastapi.routing import APIWebSocketRoute  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        websockets: dict[str, Callable[..., Any]] = {}
        for route in app.routes:
            if isinstance(route, APIWebSocketRoute):
                websockets[route.path_format] = route.endpoint
            elif hasattr(route, "effective_route_contexts"):
                for context in route.effective_route_contexts():
                    original = context.original_route
                    if not context.methods and isinstance(original, APIWebSocketRoute):
                        websockets[original.path_format] = original.endpoint

        assert set(websockets) == self._WEBSOCKET_ROUTES
        assert (
            sorted(
                path
                for path, endpoint in websockets.items()
                if not _reaches_websocket_verification(endpoint)
            )
            == []
        )


class TestAwsCredentialRecords:
    """The cross-account credential a tenant record may declare (#154).

    The operator declares ``aws_role_arn`` on the tenant record; the server
    mints the ``ExternalId`` into its own credential record. Every
    half-configured state must fail closed: a declared role silently ignored
    would land the tenant's usage on the deployment's bill.
    """

    async def test_a_declared_role_carries_the_minted_external_id(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A role-bearing tenant verifies to a credential with the stored ExternalId.

        Ref: stdapi/tenant_keys.py:_aws_credential
             https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html
        """
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        key_id = "r" + "0" * 15
        role = "arn:aws:iam::210987654321:role/stdapi-tenant"
        key = await _declare_and_mint(tenant_backend, key_id, aws_role_arn=role)
        secret_item = await get_item("TENANT", f"secret#{key_id}")
        assert secret_item is not None
        external_id = secret_item["external_id"]
        assert isinstance(external_id, str)
        assert external_id

        tenant = await verify_tenant_key(key)
        assert tenant.aws_credential is not None
        assert tenant.aws_credential.role_arn == role
        assert tenant.aws_credential.external_id == external_id

    async def test_a_tenant_without_a_role_carries_no_credential(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain tenant record yields no AWS credential.

        Ref: stdapi/tenant_keys.py:_aws_credential
        """
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        key = await _declare_and_mint(tenant_backend, "p" + "0" * 15)
        assert (await verify_tenant_key(key)).aws_credential is None

    async def test_a_declared_role_with_the_feature_off_fails_closed(
        self, tenant_backend: SSMClient
    ) -> None:
        """A role declared while tenant_aws_credentials is off refuses the key.

        503 and loud, never a silent fallback to the deployment's account:
        the operator must align the record and the setting.

        Ref: stdapi/tenant_keys.py:_aws_credential
        """
        key = await _declare_and_mint(
            tenant_backend,
            "o" + "0" * 15,
            aws_role_arn="arn:aws:iam::210987654321:role/stdapi-tenant",
        )
        with pytest.raises(FeatureUnavailableError):
            await verify_tenant_key(key)

    async def test_a_malformed_role_arn_fails_closed(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A role attribute that is not an IAM role ARN refuses the key.

        Ref: stdapi/tenant_keys.py:_aws_credential
        """
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        key = await _declare_and_mint(
            tenant_backend, "m" + "0" * 15, aws_role_arn="not-an-arn"
        )
        with pytest.raises(FeatureUnavailableError):
            await verify_tenant_key(key)

    async def test_a_missing_external_id_fails_closed_then_backfills(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A credential record predating the feature is refused, then backfilled.

        The refusal covers the up-to-a-minute window before the reconciliation
        mints the ExternalId; afterwards the key verifies with it.

        Ref: stdapi/tenant_keys.py:_backfill_external_id
        """
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        key_id = "b" + "0" * 15
        role = "arn:aws:iam::210987654321:role/stdapi-tenant"
        key = await _declare_and_mint(tenant_backend, key_id, aws_role_arn=role)
        # Rebuild the pre-#154 record shape: same secret, no ExternalId.
        secret_item = await get_item("TENANT", f"secret#{key_id}")
        assert secret_item is not None
        await put_item(
            {
                name: value
                for name, value in secret_item.items()
                if name != "external_id"
            }
        )
        tenant_keys._CACHE.clear()  # noqa: SLF001

        with pytest.raises(FeatureUnavailableError):
            await verify_tenant_key(key)

        await reconcile_tenant_keys()
        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant = await verify_tenant_key(key)
        assert tenant.aws_credential is not None
        assert tenant.aws_credential.role_arn == role
        backfilled = await get_item("TENANT", f"secret#{key_id}")
        assert backfilled is not None
        assert tenant.aws_credential.external_id == backfilled["external_id"]

    async def test_the_backfill_never_rewrites_an_existing_external_id(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reconciling again keeps the ExternalId a tenant already trusts.

        A rewritten value would break the trust policy the tenant wrote it
        into, revoking the credential from the outside.

        Ref: stdapi/tenant_keys.py:_backfill_external_id
        """
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        key_id = "s" + "0" * 15
        await _declare_and_mint(
            tenant_backend,
            key_id,
            aws_role_arn="arn:aws:iam::210987654321:role/stdapi-tenant",
        )
        before = await get_item("TENANT", f"secret#{key_id}")
        assert before is not None
        await reconcile_tenant_keys()
        after = await get_item("TENANT", f"secret#{key_id}")
        assert after is not None
        assert after["external_id"] == before["external_id"]

    async def test_the_minted_external_id_never_reaches_the_log(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both mint paths report the event without disclosing the value.

        The log is shipped to CloudWatch and read far more widely than the
        table, so the value the tenant's trust policy is conditioned on stays
        in the credential record the operator reads it from. The event itself
        is still reported, so the operator knows the ExternalId exists.

        Ref: stdapi/tenant_keys.py:_mint
             stdapi/tenant_keys.py:_backfill_external_id
        """
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        logged: list[object] = []

        def _spy(
            *detail: object, level: str | None = None, status: int | None = None
        ) -> None:
            logged.extend(detail)
            log_error_details(*detail, level=level, status=status)  # type: ignore[arg-type]

        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        monkeypatch.setattr(tenant_keys, "log_error_details", _spy)
        key_id = "l" + "0" * 15
        await _declare_and_mint(
            tenant_backend,
            key_id,
            aws_role_arn="arn:aws:iam::210987654321:role/stdapi-tenant",
        )
        minted = await get_item("TENANT", f"secret#{key_id}")
        assert minted is not None
        # Strip it back to the pre-#154 shape so the backfill mints a second one.
        await put_item(
            {name: value for name, value in minted.items() if name != "external_id"}
        )
        await reconcile_tenant_keys()
        backfilled = await get_item("TENANT", f"secret#{key_id}")
        assert backfilled is not None

        emitted = "\n".join(str(detail) for detail in logged)
        assert str(minted["external_id"]) not in emitted
        assert str(backfilled["external_id"]) not in emitted
        assert f"Minted tenant API key '{key_id}'" in emitted
        assert f"Minted the ExternalId of tenant key '{key_id}'" in emitted


class TestReconciliationLifecycle:
    """The background loop, and the failures that must never reach a caller.

    Minting runs off the request path, so every failure in it is reported and
    absorbed: a loop that dies, or a startup that fails, would leave newly
    declared tenants without a key for the life of the process.

    Ref: stdapi/tenant_keys.py:_reconcile_loop
         stdapi/tenant_keys.py:initialize_tenant_keys
    """

    async def test_the_loop_starts_once_and_stops_on_close(
        self, tenant_backend: SSMClient
    ) -> None:
        """A second start joins the running loop instead of doubling the writes."""
        del tenant_backend
        open_tenant_key_reconciliation()
        task = tenant_keys._RECONCILE_TASK  # noqa: SLF001
        try:
            open_tenant_key_reconciliation()
            assert task is not None
            assert tenant_keys._RECONCILE_TASK is task  # noqa: SLF001
        finally:
            await close_tenant_key_reconciliation()

        assert task is not None
        assert task.cancelled()
        assert tenant_keys._RECONCILE_TASK is None  # noqa: SLF001

    async def test_the_loop_survives_a_failing_pass(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreachable table pauses minting, it does not end it."""
        del tenant_backend
        passes = 0
        second = Event()

        async def _failing() -> None:
            nonlocal passes
            passes += 1
            if passes >= 2:
                second.set()
            detail = "the table is unreachable"
            raise TableUnavailableError(detail)

        monkeypatch.setattr(tenant_keys, "_RECONCILE_INTERVAL", 0.0)
        monkeypatch.setattr(tenant_keys, "reconcile_tenant_keys", _failing)
        open_tenant_key_reconciliation()
        try:
            await wait_for(second.wait(), 5.0)
            task = tenant_keys._RECONCILE_TASK  # noqa: SLF001
            assert task is not None
            assert not task.done()
        finally:
            await close_tenant_key_reconciliation()

    async def test_a_refused_delivery_is_reported_and_never_raised(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing Parameter Store permission leaves a tenant keyless, loudly.

        AWS's own message here carries no IAM denial grammar to attribute, so
        the report falls back to naming the call and the bare error code.

        Ref: stdapi/tenant_keys.py:_mint_failure_detail
        """
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        warnings: list[object] = []

        def _spy(
            *detail: object, level: str | None = None, status: int | None = None
        ) -> None:
            if level == "warning":
                warnings.extend(detail)
            log_error_details(*detail, level=level, status=status)  # type: ignore[arg-type]

        async def _denied(**_kwargs: object) -> None:
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
                "PutParameter",
            )

        monkeypatch.setattr(tenant_keys, "log_error_details", _spy)
        monkeypatch.setattr(tenant_backend, "put_parameter", _denied)
        await put_item(_tenant_item("k" + "1" * 15))

        await reconcile_tenant_keys()

        assert await get_item("TENANT", "secret#k" + "1" * 15) is None
        assert any("could not be minted" in str(warning) for warning in warnings)
        assert any(
            "PutParameter" in str(warning) and "AccessDeniedException" in str(warning)
            for warning in warnings
        )

    async def test_a_denied_delivery_names_the_missing_iam_permission(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denial IAM itself worded names the permission, never the principal.

        Reuses :func:`stdapi.api_errors.iam_denial_detail`, so the message
        names the specific action AWS refused -- here the KMS permission the
        call actually needed, not a guess between the SSM call and its key --
        while the principal ARN AWS's own message carries is never repeated.

        Ref: stdapi/tenant_keys.py:_mint_failure_detail
             stdapi/api_errors.py:iam_denial_detail
        """
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        warnings: list[object] = []

        def _spy(
            *detail: object, level: str | None = None, status: int | None = None
        ) -> None:
            if level == "warning":
                warnings.extend(detail)
            log_error_details(*detail, level=level, status=status)  # type: ignore[arg-type]

        async def _denied(**_kwargs: object) -> None:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": (
                            "User: arn:aws:sts::123456789012:assumed-role/"
                            "stdapi-server/session is not authorized to "
                            "perform: kms:GenerateDataKey on resource: "
                            "arn:aws:kms:us-east-1:123456789012:key/"
                            "12345678-1234-1234-1234-123456789012 because no "
                            "identity-based policy allows the "
                            "kms:GenerateDataKey action"
                        ),
                    }
                },
                "PutParameter",
            )

        monkeypatch.setattr(tenant_keys, "log_error_details", _spy)
        monkeypatch.setattr(tenant_backend, "put_parameter", _denied)
        await put_item(_tenant_item("k" + "2" * 15))

        await reconcile_tenant_keys()

        report = " ".join(str(warning) for warning in warnings)
        assert "kms:GenerateDataKey" in report
        assert "assumed-role/stdapi-server" not in report
        assert "is not authorized to perform" not in report

    async def test_an_unreachable_table_at_startup_only_warns(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A table a moment away from existing must not turn into an outage."""
        del tenant_backend

        async def _unreachable() -> None:
            detail = "the table is unreachable"
            raise TableUnavailableError(detail)

        monkeypatch.setattr(tenant_keys, "reconcile_tenant_keys", _unreachable)
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await initialize_tenant_keys(start_event)

        assert "server_warnings" in start_event


class TestCrossRegionKmsWarning:
    """The startup check that a delivery key encrypts from the serving region.

    Parameter Store is regional and cannot encrypt a parameter with a KMS key
    from another region, so a foreign-region key is caught by comparing the
    region an ARN carries -- never by an AWS call, since the mismatch is a
    pure config fact.

    Ref: stdapi/tenant_keys.py:initialize_tenant_keys
         stdapi/config.py:_KMS_KEY_ID_RE
    """

    @staticmethod
    async def _initialize(
        tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch, key_id: str | None
    ) -> EventLog:
        """Run startup with *key_id* configured, reconciliation stubbed to a no-op.

        Args:
            tenant_backend: The Parameter Store stand-in, only used to enable
                the feature; reconciliation itself is never exercised here.
            monkeypatch: Patches the delivery key setting and the region.
            key_id: The delivery key reference to configure.

        Returns:
            The startup event log the check reports into.
        """
        del tenant_backend

        async def _noop() -> None:
            return None

        monkeypatch.setattr(tenant_keys, "reconcile_tenant_keys", _noop)
        monkeypatch.setattr(tenant_keys, "_SSM_REGION", "us-east-1")
        monkeypatch.setattr(SETTINGS, "tenant_key_ssm_kms_key_id", key_id)
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]
        await initialize_tenant_keys(start_event)
        return start_event

    async def test_a_foreign_region_key_arn_warns_and_names_the_region(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key ARN encrypting from another region is refused before any mint."""
        key_id = (
            "arn:aws:kms:us-west-2:123456789012:key/"
            "12345678-1234-1234-1234-123456789012"
        )

        start_event = await self._initialize(tenant_backend, monkeypatch, key_id)

        warnings = start_event.get("server_warnings", [])
        assert start_event["level"] == "warning"
        assert any("us-west-2" in str(warning) for warning in warnings)

    async def test_a_same_region_key_arn_does_not_warn(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key ARN already in the serving region raises no concern."""
        key_id = (
            "arn:aws:kms:us-east-1:123456789012:key/"
            "12345678-1234-1234-1234-123456789012"
        )

        start_event = await self._initialize(tenant_backend, monkeypatch, key_id)

        assert "server_warnings" not in start_event

    async def test_a_foreign_region_alias_arn_warns_too(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An alias ARN carries a region field exactly like a key ARN does.

        Unlike a bare key id or a bare alias, an alias ARN starts with
        'arn:' and so is compared like any other ARN -- it is not exempt.
        """
        key_id = "arn:aws:kms:us-west-2:123456789012:alias/stdapi-ai"

        start_event = await self._initialize(tenant_backend, monkeypatch, key_id)

        warnings = start_event.get("server_warnings", [])
        assert any("us-west-2" in str(warning) for warning in warnings)

    @pytest.mark.parametrize(
        "key_id",
        [
            "12345678-1234-1234-1234-123456789012",
            "alias/stdapi-ai",
            "arn:aws:kms:us-east-1:123456789012:alias/stdapi-ai",
        ],
        ids=["bare-key-id", "bare-alias", "same-region-alias-arn"],
    )
    async def test_shapes_without_a_foreign_region_never_warn_or_crash(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch, key_id: str
    ) -> None:
        """A bare id or alias skips the comparison; it never starts with 'arn:'.

        ``key_id.split(":")[3]`` is only reached once *key_id* starts with
        'arn:', and every shape :data:`stdapi.config._KMS_KEY_ID_RE` accepts
        that does guarantees a region at index 3 -- so neither a missing
        index nor a false warning is possible for any accepted shape.
        """
        start_event = await self._initialize(tenant_backend, monkeypatch, key_id)

        assert "server_warnings" not in start_event

    async def test_no_delivery_key_never_warns(
        self, tenant_backend: SSMClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default, AWS-managed key raises no region concern at all."""
        start_event = await self._initialize(tenant_backend, monkeypatch, None)

        assert "server_warnings" not in start_event


@pytest.mark.gateway("Amazon DynamoDB has no upstream-vendor equivalent")
@pytest.mark.xdist_group("dynamodb")
class TestRealBackends:
    """The one behaviour the stand-ins can get subtly wrong, against AWS itself.

    Parameter Store's create-once write (``Overwrite=False``) is the mint's
    idempotency lock; the offline stand-in agrees, but the real service is the
    authority.
    """

    async def test_racing_mints_deliver_exactly_one_key(
        self, sandbox_dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Four concurrent mints against the real services agree on one secret.

        The gateway's own mint path runs here, not a re-implementation of it:
        the create-once parameter write, the ``ParameterAlreadyExists``
        recovery that reads the winner's value back, and the conditional
        credential write. What the stand-ins cannot prove is that Parameter
        Store really settles the race, and that every loser then records the
        hash of the key the winner delivered.

        Ref: stdapi/tenant_keys.py:_mint
             https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_PutParameter.html
        """
        from secrets import token_hex  # noqa: PLC0415

        from stdapi.aws import _CLIENTS  # noqa: PLC0415
        from stdapi.config import AWS_REGION  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", sandbox_dynamodb_table)
        monkeypatch.setattr(SETTINGS, "tenant_key_ssm_parameter_prefix", _PREFIX)
        key_id = token_hex(8)
        parameter = f"{_PREFIX}/{key_id}"
        session = get_session()
        async with session.create_client("ssm", region_name=AWS_REGION) as ssm_client:
            monkeypatch.setitem(_CLIENTS, "ssm", {AWS_REGION: ssm_client})
            try:
                await gather(
                    *(tenant_keys._mint(key_id, "race") for _ in range(4))  # noqa: SLF001
                )

                delivered = await ssm_client.get_parameter(
                    Name=parameter, WithDecryption=True
                )
                assert delivered["Parameter"]["Version"] == 1
                item = await get_item("TENANT", f"secret#{key_id}")
                assert item is not None
                salt = item["salt"]
                assert isinstance(salt, bytes)
                secret = delivered["Parameter"]["Value"].rsplit("-", 1)[-1]
                assert item["secret_hash"] == tenant_keys._hash_secret(secret, salt)  # noqa: SLF001
            finally:
                await ssm_client.delete_parameter(Name=parameter)
                await delete_item("TENANT", f"secret#{key_id}")
