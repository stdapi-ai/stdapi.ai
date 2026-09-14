"""Tests for tenant API keys stored in, and rotated through, AWS Secrets Manager.

With ``tenant_key_secretsmanager_prefix`` set, each tenant's key lives in a
secret of its own instead of a one-shot Parameter Store delivery, and the
server rotates it in place through the native staging labels: the new key is
written as ``AWSPENDING``, recorded in the table, promoted to ``AWSCURRENT``
and the superseded one is kept readable as ``AWSPREVIOUS``. Both keys
authenticate for an overlap window so a client that has not re-read the
secret yet is not locked out.

The offline lane runs against moto's server mode. Three things it gets
differently from the real service, recorded here so a green run is not read
as proof of them: it accepts a reused ``ClientRequestToken`` carrying a
different value (AWS refuses it), its reads never lag its writes (AWS's do,
by up to a second), and a read naming no stage answers the version written
last rather than ``AWSCURRENT``. The first two paths are exercised by
injecting the errors the real service answers with; every read here names
its stage.

Ref: stdapi/tenant_keys.py
     https://docs.aws.amazon.com/secretsmanager/latest/userguide/getting-started.html#term_version
     https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_UpdateSecretVersionStage.html
"""

from __future__ import annotations

from contextlib import suppress
from hmac import compare_digest
from time import time
from typing import TYPE_CHECKING, Any

import pytest
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from pydantic import ValidationError

from stdapi import tenant_keys
from stdapi.api_errors import ApiError
from stdapi.aws_dynamodb import PARTITION_KEY, SORT_KEY, delete_item, get_item, put_item
from stdapi.config import SETTINGS, _Settings
from stdapi.monitoring import TENANT, EventLog
from stdapi.tenant_keys import (
    KEY_PREFIX,
    initialize_tenant_keys,
    reconcile_tenant_keys,
    tenant_key_client_specs,
    verify_tenant_key,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from stdapi.aws_dynamodb import Item, ItemValue

#: The Secrets Manager stand-in client; no stub package types it in this environment.
SecretsManagerClient = Any

#: Secrets Manager prefix the offline lane stores keys under.
_PREFIX = "stdapi-test/tenant-keys"

#: A well-formed key ID that no record backs.
_UNKNOWN_KEY_ID = "A" * 16

#: A well-formed secret of the minted length.
_WELL_FORMED_SECRET = "s" * 43

#: Seconds in a day, the unit of the rotation schedule.
_DAY = 86400


def _tenant_item(key_id: str, **attributes: ItemValue) -> Item:
    """Build the tenant record the Terraform module would write.

    Args:
        key_id: The tenant's key ID.
        **attributes: Extra record attributes, e.g. ``key_generation``.

    Returns:
        The item.
    """
    return {
        PARTITION_KEY: "TENANT",
        SORT_KEY: f"tenant#{key_id}",
        "name": f"tenant-{key_id}",
        **attributes,
    }


def _settings(**overrides: Any) -> _Settings:  # noqa: ANN401 - settings kwargs
    """Build a settings model with the Secrets Manager store enabled.

    Args:
        **overrides: Fields overriding the enabled-store baseline.

    Returns:
        The validated settings.
    """
    return _Settings(
        **{
            "tenant_api_keys": True,
            "aws_dynamodb_table": "shared",
            "tenant_key_secretsmanager_prefix": _PREFIX,
            **overrides,
        }
    )


@pytest.fixture
async def secrets_backend(
    dynamodb_table: str,  # noqa: ARG001 - binds the table stand-in
    moto_dynamodb_endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[SecretsManagerClient]:
    """Enable tenant keys against the local table and Secrets Manager stand-ins.

    Yields:
        The Secrets Manager client bound to the stand-in, for reading stored
        keys back and for injecting the errors moto cannot produce.

    Ref: tests/conftest.py:dynamodb_table
         tests/test_tenant_keys.py:tenant_backend
    """
    from stdapi.aws import _CLIENTS  # noqa: PLC0415

    region = SETTINGS.aws_bedrock_regions[0]
    session = get_session()
    async with session.create_client(
        "secretsmanager",
        region_name=region,
        endpoint_url=moto_dynamodb_endpoint,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # noqa: S106 - a local stand-in, not a secret
    ) as client:
        monkeypatch.setitem(_CLIENTS, "secretsmanager", {region: client})
        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        monkeypatch.setattr(SETTINGS, "tenant_key_cache_seconds", 60.0)
        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_prefix", _PREFIX)
        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_kms_key_id", None)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_days", None)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", 604800)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant_keys._NEGATIVE.clear()  # noqa: SLF001
        tenant_keys._REPORTED.clear()  # noqa: SLF001
        token = TENANT.set(None)
        try:
            yield client
        finally:
            TENANT.reset(token)
            tenant_keys._CACHE.clear()  # noqa: SLF001
            tenant_keys._NEGATIVE.clear()  # noqa: SLF001
            tenant_keys._REPORTED.clear()  # noqa: SLF001


async def _current_key(client: SecretsManagerClient, key_id: str) -> str:
    """Read the key a tenant's secret currently holds.

    Args:
        client: The Secrets Manager stand-in client.
        key_id: The tenant's key ID.

    Returns:
        The full API key, as a client re-reading the secret would get it.
    """
    value: str = (
        await client.get_secret_value(
            SecretId=f"{_PREFIX}/{key_id}", VersionStage="AWSCURRENT"
        )
    )["SecretString"]
    assert value.startswith(f"{KEY_PREFIX}{key_id}-")
    return value


async def _pending_key(client: SecretsManagerClient, key_id: str) -> str:
    """Read the key a tenant's secret holds as its pending version.

    Args:
        client: The Secrets Manager stand-in client.
        key_id: The tenant's key ID.

    Returns:
        The full API key a rotation staged but has not promoted.
    """
    value: str = (
        await client.get_secret_value(
            SecretId=f"{_PREFIX}/{key_id}", VersionStage="AWSPENDING"
        )
    )["SecretString"]
    assert value.startswith(f"{KEY_PREFIX}{key_id}-")
    return value


async def _declare_and_mint(
    client: SecretsManagerClient, key_id: str = "k" + "0" * 15, **attributes: ItemValue
) -> str:
    """Declare one tenant, run the reconciliation, and read back its key.

    Args:
        client: The Secrets Manager stand-in client.
        key_id: The tenant's key ID.
        **attributes: Extra tenant-record attributes.

    Returns:
        The full API key, as the tenant reads it from its secret.
    """
    await put_item(_tenant_item(key_id, **attributes))
    await reconcile_tenant_keys()
    return await _current_key(client, key_id)


async def _secret_record(key_id: str) -> Item:
    """Read a tenant's credential record, asserting it exists.

    Args:
        key_id: The tenant's key ID.

    Returns:
        The record.
    """
    item = await get_item("TENANT", f"secret#{key_id}")
    assert item is not None
    return item


async def _age_record(key_id: str, seconds: int) -> None:
    """Push a credential record's mint or rotation time *seconds* into the past.

    Args:
        key_id: The tenant's key ID.
        seconds: How far back to move the record's last rotation.
    """
    item = await _secret_record(key_id)
    last = "rotated_at" if "rotated_at" in item else "minted_at"
    stamp = item[last]
    assert isinstance(stamp, int)
    await put_item({**item, last: stamp - seconds})


async def _rotate_once(client: SecretsManagerClient, key_id: str) -> tuple[str, str]:
    """Mint a tenant and rotate its key once, on demand.

    Args:
        client: The Secrets Manager stand-in client.
        key_id: The tenant's key ID.

    Returns:
        The key before the rotation and the key after it.
    """
    old = await _declare_and_mint(client, key_id=key_id)
    await put_item(_tenant_item(key_id, key_generation=1))
    await reconcile_tenant_keys()
    new = await _current_key(client, key_id)
    assert new != old
    return old, new


def _client_error(code: str, operation: str) -> ClientError:
    """Build the error a Secrets Manager call answers with.

    Args:
        code: The AWS error code.
        operation: The operation that failed.

    Returns:
        The error.
    """
    return ClientError({"Error": {"Code": code, "Message": "injected"}}, operation)


def _spy_compares(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bytes, bytes]]:
    """Record every constant-time comparison the verifier makes.

    Args:
        monkeypatch: The patcher restoring the comparison afterwards.

    Returns:
        The operands of each comparison, in order.
    """
    compared: list[tuple[bytes, bytes]] = []

    def _spy(left: bytes, right: bytes) -> bool:
        compared.append((left, right))
        return compare_digest(left, right)

    monkeypatch.setattr(tenant_keys, "compare_digest", _spy)
    return compared


def _spy_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every table read the verifier makes.

    Args:
        monkeypatch: The patcher restoring the read afterwards.

    Returns:
        The sort key of each read, in order.
    """
    reads: list[str] = []

    async def _spy(
        partition_key: str, sort_key: str, *, consistent: bool = False
    ) -> Item | None:
        reads.append(sort_key)
        return await get_item(partition_key, sort_key, consistent=consistent)

    monkeypatch.setattr(tenant_keys, "get_item", _spy)
    return reads


def _spy_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every warning the module reports to the operator.

    Args:
        monkeypatch: The patcher restoring the logger afterwards.

    Returns:
        The warning details, in order.
    """
    from stdapi.monitoring import log_error_details  # noqa: PLC0415

    warnings: list[str] = []

    def _spy(
        *detail: object, level: str | None = None, status: int | None = None
    ) -> None:
        if level == "warning":
            warnings.extend(str(item) for item in detail)
        log_error_details(*detail, level=level, status=status)  # type: ignore[arg-type]

    monkeypatch.setattr(tenant_keys, "log_error_details", _spy)
    return warnings


def _deny_stage_moves(
    client: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse every stage move, as a role without the permission would.

    ``secretsmanager:UpdateSecretVersionStage`` is the one permission the
    mint path never exercises, so a policy missing it first fails at a
    rotation, possibly months after the deployment.

    Args:
        client: The Secrets Manager stand-in client.
        monkeypatch: The patcher restoring the client afterwards.
    """
    error = _client_error("AccessDeniedException", "UpdateSecretVersionStage")

    async def _denied(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(client, "update_secret_version_stage", _denied)


def _frozen_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the module's wall clock with one the test moves by hand.

    Args:
        monkeypatch: The patcher restoring the clock afterwards.

    Returns:
        A one-element list holding the epoch second every ``time()`` call in
        the module answers; assign to its first element to move the clock.
    """
    clock = [float(int(time()))]
    monkeypatch.setattr(tenant_keys, "time", lambda: clock[0])
    return clock


def _record_calls(
    client: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch, operation: str
) -> list[dict[str, Any]]:
    """Record the requests of one Secrets Manager operation, and send them on.

    Args:
        client: The Secrets Manager stand-in client.
        monkeypatch: The patcher restoring the client afterwards.
        operation: The client method to record, e.g. ``create_secret``.

    Returns:
        The keyword arguments of each call, in order.
    """
    requests: list[dict[str, Any]] = []
    original = getattr(client, operation)

    async def _record(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
        requests.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(client, operation, _record)
    return requests


def _fail_once(
    client: SecretsManagerClient,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    code: str,
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Make the first call of one operation fail with *code*, then pass through.

    Args:
        client: The Secrets Manager stand-in client.
        monkeypatch: The patcher restoring the client afterwards.
        operation: The client method to fail, e.g. ``put_secret_value``.
        code: The AWS error code to answer with.

    Returns:
        The stand-in, whose ``__name__`` is the operation for the log.
    """
    original = getattr(client, operation)
    failed = False

    async def _once(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
        nonlocal failed
        if not failed:
            failed = True
            raise _client_error(code, operation)
        return await original(**kwargs)

    monkeypatch.setattr(client, operation, _once)
    return _once


class TestRotationConfiguration:
    """What the settings model accepts for the store and the schedule.

    Ref: stdapi/config.py:_validate_tenant_keys
         https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
    """

    def test_the_store_alone_is_a_complete_configuration(self) -> None:
        """The prefix selects the store; rotation stays on demand until scheduled.

        Ref: stdapi/config.py:_validate_tenant_keys
        """
        settings = _settings()

        assert settings.tenant_key_secretsmanager_prefix == _PREFIX
        assert settings.tenant_key_rotation_days is None
        assert settings.tenant_key_rotation_overlap_seconds == 604800

    def test_a_trailing_slash_on_the_prefix_is_dropped(self) -> None:
        """The secret name is ``<prefix>/<key id>``, never with a double slash.

        Ref: stdapi/config.py:_validate_tenant_keys
        """
        settings = _settings(tenant_key_secretsmanager_prefix=f"{_PREFIX}/")

        assert settings.tenant_key_secretsmanager_prefix == _PREFIX

    @pytest.mark.parametrize(
        "prefix",
        ["/leading/slash", "with space", "semi;colon", "", "a" * 500],
        ids=["leading-slash", "space", "punctuation", "empty", "too-long"],
    )
    def test_a_prefix_that_is_not_a_secret_name_fails_startup(
        self, prefix: str
    ) -> None:
        """Secrets Manager names take letters, digits and ``/_+=.@-`` up to 512.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
        """
        with pytest.raises(ValidationError, match="tenant_key_secretsmanager_prefix"):
            _settings(tenant_key_secretsmanager_prefix=prefix)

    def test_the_store_without_the_feature_fails_startup(self) -> None:
        """Nothing is ever stored under it, so the setting is a mistake.

        Ref: stdapi/config.py:_validate_tenant_keys
        """
        with pytest.raises(ValidationError, match="requires tenant_api_keys"):
            _Settings(tenant_key_secretsmanager_prefix=_PREFIX)

    def test_a_schedule_without_the_store_fails_startup(self) -> None:
        """A rotation has nowhere to publish the new key without the store.

        Ref: stdapi/config.py:_validate_tenant_keys
        """
        with pytest.raises(
            ValidationError, match="requires tenant_key_secretsmanager_prefix"
        ):
            _Settings(
                tenant_api_keys=True,
                aws_dynamodb_table="shared",
                tenant_key_rotation_days=30,
            )

    def test_a_store_key_without_the_store_fails_startup(self) -> None:
        """The encryption key of a store that is not used is a mistake.

        Ref: stdapi/config.py:_validate_tenant_keys
        """
        with pytest.raises(
            ValidationError, match="requires tenant_key_secretsmanager_prefix"
        ):
            _Settings(
                tenant_api_keys=True,
                aws_dynamodb_table="shared",
                tenant_key_secretsmanager_kms_key_id="alias/stdapi-ai",
            )

    @pytest.mark.parametrize("days", [0, -1])
    def test_a_schedule_shorter_than_a_day_fails_startup(self, days: int) -> None:
        """The schedule is a number of whole days, at least one.

        Ref: stdapi/config.py:_Settings.tenant_key_rotation_days
        """
        with pytest.raises(ValidationError, match="tenant_key_rotation_days"):
            _settings(tenant_key_rotation_days=days)

    def test_a_negative_overlap_fails_startup(self) -> None:
        """The overlap is a duration; zero is the hard cutover, less is nonsense.

        Ref: stdapi/config.py:_Settings.tenant_key_rotation_overlap_seconds
        """
        with pytest.raises(
            ValidationError, match="tenant_key_rotation_overlap_seconds"
        ):
            _settings(tenant_key_rotation_overlap_seconds=-1)

    def test_an_overlap_reaching_the_next_rotation_fails_startup(self) -> None:
        """An overlap no shorter than the schedule retires nothing.

        Only the last superseded key is kept, so the grace ends at the next
        rotation whatever the setting says: the two are only meaningful
        together, and the combination that cancels the rotation is refused
        rather than left to look configured.

        Ref: stdapi/config.py:_Settings._validate_tenant_key_store
        """
        with pytest.raises(
            ValidationError, match="must be shorter than tenant_key_rotation_days"
        ):
            _settings(
                tenant_key_rotation_days=1, tenant_key_rotation_overlap_seconds=_DAY
            )

    def test_an_overlap_inside_the_rotation_schedule_is_accepted(self) -> None:
        """The ordinary pairing is left alone.

        Ref: stdapi/config.py:_Settings._validate_tenant_key_store
        """
        settings = _settings(
            tenant_key_rotation_days=30, tenant_key_rotation_overlap_seconds=_DAY
        )

        assert settings.tenant_key_rotation_overlap_seconds == _DAY

    def test_a_malformed_store_key_reference_fails_startup(self) -> None:
        """A typo would otherwise only surface when the first tenant is minted.

        Ref: stdapi/config.py:_KMS_KEY_ID_RE
        """
        with pytest.raises(
            ValidationError, match="tenant_key_secretsmanager_kms_key_id"
        ):
            _settings(tenant_key_secretsmanager_kms_key_id="not a key")

    def test_the_store_selects_the_secrets_manager_client_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the store on, no Parameter Store client is opened, and vice versa.

        Ref: stdapi/tenant_keys.py:tenant_key_client_specs
        """
        monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_prefix", _PREFIX)
        assert [service for service, _ in tenant_key_client_specs()] == [
            "secretsmanager"
        ]

        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_prefix", None)
        assert [service for service, _ in tenant_key_client_specs()] == ["ssm"]


class TestStoreKmsWarning:
    """The startup check that the store's key encrypts from the serving region.

    Ref: stdapi/tenant_keys.py:initialize_tenant_keys
         tests/test_tenant_keys.py:TestCrossRegionKmsWarning
    """

    @staticmethod
    async def _initialize(
        monkeypatch: pytest.MonkeyPatch, key_id: str | None
    ) -> EventLog:
        """Run startup with *key_id* as the store key, reconciliation stubbed.

        Args:
            monkeypatch: Patches the store key setting and the region.
            key_id: The store key reference to configure.

        Returns:
            The startup event log the check reports into.
        """

        async def _noop() -> dict[str, Item]:
            return {}

        monkeypatch.setattr(tenant_keys, "reconcile_tenant_keys", _noop)
        monkeypatch.setattr(tenant_keys, "_STORE_REGION", "us-east-1")
        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_kms_key_id", key_id)
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]
        await initialize_tenant_keys(start_event)
        return start_event

    async def test_a_foreign_region_key_arn_warns_and_names_the_region(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key ARN encrypting from another region is caught before any mint."""
        del secrets_backend
        key_id = (
            "arn:aws:kms:us-west-2:123456789012:key/"
            "12345678-1234-1234-1234-123456789012"
        )

        start_event = await self._initialize(monkeypatch, key_id)

        warnings = start_event.get("server_warnings", [])
        assert any("us-west-2" in str(warning) for warning in warnings)

    async def test_a_same_region_key_arn_does_not_warn(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key ARN already in the serving region raises no concern."""
        del secrets_backend
        key_id = (
            "arn:aws:kms:us-east-1:123456789012:key/"
            "12345678-1234-1234-1234-123456789012"
        )

        start_event = await self._initialize(monkeypatch, key_id)

        assert "server_warnings" not in start_event


class TestMintingIntoTheStore:
    """A declared tenant gets its key stored as the secret's current version.

    Ref: stdapi/tenant_keys.py:_store_key
         https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
    """

    async def test_a_pending_tenant_is_minted_and_its_key_validates(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """The secret's current version holds a key the gateway then accepts.

        Ref: stdapi/tenant_keys.py:_store_key
        """
        key = await _declare_and_mint(secrets_backend)

        tenant = await verify_tenant_key(key)

        assert tenant.key_id == "k" + "0" * 15
        item = await _secret_record("k" + "0" * 15)
        secret = key.rsplit("-", 1)[-1]
        assert secret not in str(item)

    async def test_no_parameter_store_delivery_happens(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """With the store on, the one-shot Parameter Store delivery is not used.

        The pool holds no Parameter Store client at all on this path, so a
        delivery attempt would fail rather than silently duplicate the key.

        Ref: stdapi/tenant_keys.py:tenant_key_client_specs
        """
        from stdapi.aws import _CLIENTS  # noqa: PLC0415

        await _declare_and_mint(secrets_backend, key_id="p" + "0" * 15)

        assert "ssm" not in _CLIENTS

    async def test_the_configured_kms_key_encrypts_the_secret(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The setting reaches Secrets Manager as the secret's KmsKeyId.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
        """
        monkeypatch.setattr(
            SETTINGS, "tenant_key_secretsmanager_kms_key_id", "alias/stdapi-ai"
        )
        requests = _record_calls(secrets_backend, monkeypatch, "create_secret")

        await _declare_and_mint(secrets_backend, key_id="m" + "0" * 15)

        assert requests[0]["KmsKeyId"] == "alias/stdapi-ai"
        assert requests[0]["Name"] == f"{_PREFIX}/m{'0' * 15}"

    async def test_no_configured_kms_key_leaves_the_field_out(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset sends no KmsKeyId, which selects the AWS-managed key.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
        """
        requests = _record_calls(secrets_backend, monkeypatch, "create_secret")

        await _declare_and_mint(secrets_backend, key_id="n" + "0" * 15)

        assert "KmsKeyId" not in requests[0]

    async def test_minting_is_idempotent_across_runs(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A second reconciliation changes neither the secret nor the hash.

        Ref: stdapi/tenant_keys.py:_store_key
        """
        key = await _declare_and_mint(secrets_backend, key_id="i" + "0" * 15)
        before = await _secret_record("i" + "0" * 15)

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, "i" + "0" * 15) == key
        assert await _secret_record("i" + "0" * 15) == before

    async def test_a_crash_between_storing_and_recording_recovers(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """The secret is the source of truth: the hash is rebuilt from it.

        Ref: stdapi/tenant_keys.py:_store_key
        """
        key = await _declare_and_mint(secrets_backend, key_id="c" + "0" * 15)
        await delete_item("TENANT", "secret#c" + "0" * 15)

        await reconcile_tenant_keys()

        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant_keys._NEGATIVE.clear()  # noqa: SLF001
        assert (await verify_tenant_key(key)).key_id == "c" + "0" * 15
        assert await _current_key(secrets_backend, "c" + "0" * 15) == key

    async def test_an_empty_container_created_by_tooling_receives_the_key(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A secret the Terraform module pre-created gets its first version.

        The module owns the container (tags, encryption key, lifecycle) and
        the server owns the versions, so a container without any version is
        the expected state of a freshly declared tenant.

        Ref: stdapi/tenant_keys.py:_store_key
             https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_PutSecretValue.html
        """
        key_id = "e" + "0" * 15
        await secrets_backend.create_secret(Name=f"{_PREFIX}/{key_id}")
        creates = _record_calls(secrets_backend, monkeypatch, "create_secret")

        key = await _declare_and_mint(secrets_backend, key_id=key_id)

        assert (await verify_tenant_key(key)).key_id == key_id
        assert len(creates) == 1, "the container is tried first, then adopted"

    async def test_a_secret_holding_something_else_is_never_recorded(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator-planted value cannot become a tenant's credential.

        The refusal names the secret for the operator, and never its value.

        Ref: stdapi/tenant_keys.py:_store_key
        """
        key_id = "j" + "0" * 15
        planted = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"
        await secrets_backend.create_secret(
            Name=f"{_PREFIX}/{key_id}", SecretString=planted
        )
        warnings = _spy_warnings(monkeypatch)
        await put_item(_tenant_item(key_id))

        await reconcile_tenant_keys()

        assert await get_item("TENANT", f"secret#{key_id}") is None
        assert any(f"{_PREFIX}/{key_id}" in warning for warning in warnings)
        assert not any(planted in warning for warning in warnings)

    async def test_racing_mints_converge_on_the_version_that_landed(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loser of the create race records the hash of the winner's key.

        The real service refuses a reused request token carrying a different
        value; moto does not, so the refusal is injected on top of a version
        another instance already wrote.

        Ref: stdapi/tenant_keys.py:_store_key
             https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
        """
        key_id = "r" + "0" * 15
        winner = f"{KEY_PREFIX}{key_id}-{'w' * 43}"
        await secrets_backend.create_secret(
            Name=f"{_PREFIX}/{key_id}",
            ClientRequestToken=tenant_keys._version_token(key_id, 0),  # noqa: SLF001
            SecretString=winner,
        )
        _fail_once(
            secrets_backend, monkeypatch, "create_secret", "ResourceExistsException"
        )
        await put_item(_tenant_item(key_id))

        await reconcile_tenant_keys()

        assert (await verify_tenant_key(winner)).key_id == key_id

    async def test_a_denied_store_write_names_the_missing_iam_permission(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused CreateSecret is reported with the secret and the action.

        Ref: stdapi/tenant_keys.py:_mint_failure_detail
             stdapi/api_errors.py:iam_denial_detail
        """
        warnings = _spy_warnings(monkeypatch)

        async def _denied(**_kwargs: object) -> None:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": (
                            "User: arn:aws:sts::123456789012:assumed-role/x/y is "
                            "not authorized to perform: secretsmanager:CreateSecret "
                            "on resource: arn:aws:secretsmanager:eu-west-3:"
                            f"123456789012:secret:{_PREFIX}/d{'0' * 15}-AbCdEf"
                        ),
                    }
                },
                "CreateSecret",
            )

        monkeypatch.setattr(secrets_backend, "create_secret", _denied)
        await put_item(_tenant_item("d" + "0" * 15))

        await reconcile_tenant_keys()

        assert any(
            "secretsmanager:CreateSecret" in warning and f"{_PREFIX}/d" in warning
            for warning in warnings
        )
        assert await get_item("TENANT", "secret#d" + "0" * 15) is None

    async def test_a_destroyed_tenant_keeps_its_container_for_the_operator(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The credential record is revoked; the secret is the operator's to delete.

        The container is owned by whoever declared the tenant -- the Terraform
        module destroys it with the tenant record -- so the server never
        deletes one: a deletion racing the module's own would fail its apply.
        The revocation names the secret once, so nothing is left unnoticed.

        Ref: stdapi/tenant_keys.py:_drop_orphan
        """
        key_id = "o" + "0" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        warnings = _spy_warnings(monkeypatch)
        await delete_item("TENANT", f"tenant#{key_id}")

        await reconcile_tenant_keys()

        assert await get_item("TENANT", f"secret#{key_id}") is None
        assert await _current_key(secrets_backend, key_id) == key
        assert any(f"{_PREFIX}/{key_id}" in warning for warning in warnings)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        with pytest.raises(ApiError):
            await verify_tenant_key(key)


class TestRotationTriggers:
    """When a key is rotated: on the schedule, on demand, and never twice.

    Ref: stdapi/tenant_keys.py:_rotation_due
    """

    async def test_a_key_older_than_the_schedule_is_rotated(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record past ``tenant_key_rotation_days`` gets a new key next pass.

        Ref: stdapi/tenant_keys.py:_rotation_due
        """
        key_id = "s" + "0" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_days", 30)
        await _age_record(key_id, 31 * _DAY)

        await reconcile_tenant_keys()

        new = await _current_key(secrets_backend, key_id)
        assert new != old
        previous = await secrets_backend.get_secret_value(
            SecretId=f"{_PREFIX}/{key_id}", VersionStage="AWSPREVIOUS"
        )
        assert previous["SecretString"] == old
        record = await _secret_record(key_id)
        assert record["rotations"] == 1
        assert "pending_version_id" not in record

    async def test_a_key_younger_than_the_schedule_is_left_alone(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Steady state makes no Secrets Manager call at all.

        Ref: stdapi/tenant_keys.py:_rotation_due
        """
        key_id = "y" + "0" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_days", 30)
        await _age_record(key_id, 29 * _DAY)
        puts = _record_calls(secrets_backend, monkeypatch, "put_secret_value")
        describes = _record_calls(secrets_backend, monkeypatch, "describe_secret")

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == key
        assert not puts
        assert not describes

    async def test_a_rotated_key_waits_a_full_period_again(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The schedule counts from the rotation, not from the mint.

        Ref: stdapi/tenant_keys.py:_rotation_due
        """
        key_id = "t" + "0" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_days", 30)
        await _age_record(key_id, 31 * _DAY)
        await reconcile_tenant_keys()
        rotated = await _current_key(secrets_backend, key_id)

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == rotated
        assert (await _secret_record(key_id))["rotations"] == 1

    async def test_a_generation_bump_rotates_once_and_not_again(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """``key_generation`` above the recorded generation rotates exactly once.

        Ref: stdapi/tenant_keys.py:_rotation_due
        """
        key_id = "g" + "0" * 15
        _, new = await _rotate_once(secrets_backend, key_id)

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == new
        record = await _secret_record(key_id)
        assert record["rotations"] == 1
        assert record["generation"] == 1

    async def test_a_generation_declared_at_mint_does_not_rotate(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A tenant minted at generation 3 is at generation 3, not due.

        Ref: stdapi/tenant_keys.py:_mint
        """
        key_id = "h" + "0" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id, key_generation=3)

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == key
        record = await _secret_record(key_id)
        assert record["generation"] == 3
        assert "rotations" not in record

    async def test_a_lower_generation_never_rotates_backwards(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """Lowering ``key_generation`` is not a request; only an increase is.

        Ref: stdapi/tenant_keys.py:_rotation_due
        """
        key_id = "l" + "0" * 15
        _, new = await _rotate_once(secrets_backend, key_id)
        await put_item(_tenant_item(key_id, key_generation=0))

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == new


class TestRotationProtocol:
    """The four staging-label steps, their ordering, and their retries.

    Ref: stdapi/tenant_keys.py:_rotate
         https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_UpdateSecretVersionStage.html
    """

    async def test_the_table_accepts_the_new_key_before_it_becomes_current(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client re-reading the secret is never handed a key the gateway refuses.

        Ref: stdapi/tenant_keys.py:_rotate
        """
        key_id = "q" + "0" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        before = (await _secret_record(key_id))["secret_hash"]
        hashes_at_promotion: list[object] = []
        original = secrets_backend.update_secret_version_stage

        async def _observe(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
            if kwargs.get("VersionStage") == "AWSCURRENT":
                hashes_at_promotion.append(
                    (await _secret_record(key_id))["secret_hash"]
                )
            return await original(**kwargs)

        monkeypatch.setattr(secrets_backend, "update_secret_version_stage", _observe)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert hashes_at_promotion, "the promotion must have been observed"
        assert all(digest != before for digest in hashes_at_promotion)

    async def test_the_promotion_names_the_version_it_takes_the_label_from(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Moving AWSCURRENT requires RemoveFromVersionId, so it is read first.

        Both the real service and the stand-in refuse a move without it; the
        version is read from the secret's own label map, without decrypting
        anything, immediately before the move.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_UpdateSecretVersionStage.html
             https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_DescribeSecret.html
        """
        key_id = "v" + "0" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        first = (
            await secrets_backend.get_secret_value(SecretId=f"{_PREFIX}/{key_id}")
        )["VersionId"]
        moves = _record_calls(
            secrets_backend, monkeypatch, "update_secret_version_stage"
        )
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        promotion = next(m for m in moves if m["VersionStage"] == "AWSCURRENT")
        assert promotion["RemoveFromVersionId"] == first
        assert promotion["MoveToVersionId"] == tenant_keys._version_token(key_id, 1)  # noqa: SLF001
        assert any(
            m["VersionStage"] == "AWSPENDING" and "RemoveFromVersionId" in m
            for m in moves
        ), "the pending label is cleared once the version is current"

    async def test_a_crash_after_recording_redoes_the_promotion(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A record still carrying ``pending_version_id`` is finished next pass.

        Ref: stdapi/tenant_keys.py:_rotate
        """
        key_id = "w" + "0" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        token = tenant_keys._version_token(key_id, 1)  # noqa: SLF001
        new = f"{KEY_PREFIX}{key_id}-{'n' * 43}"
        await secrets_backend.put_secret_value(
            SecretId=f"{_PREFIX}/{key_id}",
            ClientRequestToken=token,
            SecretString=new,
            VersionStages=["AWSPENDING"],
        )
        record = await _secret_record(key_id)
        salt = record["salt"]
        assert isinstance(salt, bytes)
        await put_item(
            {
                **record,
                "secret_hash": tenant_keys._hash_secret("n" * 43, salt),  # noqa: SLF001
                "rotations": 1,
                "pending_version_id": token,
            }
        )
        assert await _current_key(secrets_backend, key_id) == old

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == new
        assert "pending_version_id" not in await _secret_record(key_id)

    async def test_a_crash_before_recording_adopts_the_pending_version(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pending version already written under the token is read back.

        The retry draws a fresh secret, so its write is refused by the real
        service for reusing the token; the version that landed is adopted.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_PutSecretValue.html
        """
        key_id = "x" + "0" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        pending = f"{KEY_PREFIX}{key_id}-{'p' * 43}"
        await secrets_backend.put_secret_value(
            SecretId=f"{_PREFIX}/{key_id}",
            ClientRequestToken=tenant_keys._version_token(key_id, 1),  # noqa: SLF001
            SecretString=pending,
            VersionStages=["AWSPENDING"],
        )
        _fail_once(
            secrets_backend, monkeypatch, "put_secret_value", "ResourceExistsException"
        )
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == pending
        assert (await verify_tenant_key(pending)).key_id == key_id

    async def test_a_lagging_read_of_the_pending_version_is_retried(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The service's reads lag its writes; a first miss is not a failure.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        """
        key_id = "z" + "0" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        pending = f"{KEY_PREFIX}{key_id}-{'p' * 43}"
        await secrets_backend.put_secret_value(
            SecretId=f"{_PREFIX}/{key_id}",
            ClientRequestToken=tenant_keys._version_token(key_id, 1),  # noqa: SLF001
            SecretString=pending,
            VersionStages=["AWSPENDING"],
        )
        _fail_once(
            secrets_backend, monkeypatch, "put_secret_value", "ResourceExistsException"
        )
        _fail_once(
            secrets_backend,
            monkeypatch,
            "get_secret_value",
            "ResourceNotFoundException",
        )
        monkeypatch.setattr(tenant_keys, "_READ_LAG_SECONDS", 0.0)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == pending

    async def test_a_rotation_does_not_erase_an_externalid_minted_beside_it(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A backfill and a rotation in one pass leave the ExternalId in place.

        Both work sets are built from the same listing, and rotating re-puts
        the whole record from that copy: the ExternalId written moments before
        was overwritten, and every request from the tenant was then refused as
        unavailable until a later pass minted a new one -- a new value, which
        the role's trust policy no longer requires.

        Ref: stdapi/tenant_keys.py:reconcile_tenant_keys
        """
        key_id = "e" + "1" * 15
        record: Item = {
            PARTITION_KEY: "TENANT",
            SORT_KEY: f"secret#{key_id}",
            "secret_hash": tenant_keys._hash_secret(_WELL_FORMED_SECRET, b"\0" * 16),  # noqa: SLF001
            "salt": b"\0" * 16,
            "minted_at": 1,
        }
        await put_item(record)
        await put_item(
            _tenant_item(
                key_id,
                aws_role_arn="arn:aws:iam::210987654321:role/stdapi-tenant",
                key_generation=1,
            )
        )

        await reconcile_tenant_keys()

        assert (await _secret_record(key_id)).get("external_id"), (
            "the rotation re-put the record listed before the backfill wrote it"
        )

    async def test_a_rotation_of_a_key_without_a_secret_creates_one(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A key minted before the store was enabled is rotated into a new secret.

        Its credential record exists, its one-shot delivery is long deleted,
        and no secret holds it: the rotation creates the container and the
        new key becomes its first, current version.

        Ref: stdapi/tenant_keys.py:_put_pending
        """
        key_id = "u" + "0" * 15
        record: Item = {
            PARTITION_KEY: "TENANT",
            SORT_KEY: f"secret#{key_id}",
            "secret_hash": tenant_keys._hash_secret("o" * 43, b"\0" * 16),  # noqa: SLF001
            "salt": b"\0" * 16,
            "external_id": "legacy",
            "minted_at": 1,
        }
        await put_item(record)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        new = await _current_key(secrets_backend, key_id)
        assert (await verify_tenant_key(new)).key_id == key_id
        assert (
            await verify_tenant_key(f"{KEY_PREFIX}{key_id}-{'o' * 43}")
        ).key_id == key_id

    async def test_a_planted_pending_version_is_never_recorded(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pending version holding something else aborts the rotation.

        Whoever can write the secret could otherwise choose a tenant's next
        key: the value read back under the token must be this tenant's key,
        or nothing is recorded and the current key stays as it was.

        Ref: stdapi/tenant_keys.py:_adopt
        """
        key_id = "a" + "5" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        planted = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{'p' * 43}"
        await secrets_backend.put_secret_value(
            SecretId=f"{_PREFIX}/{key_id}",
            ClientRequestToken=tenant_keys._version_token(key_id, 1),  # noqa: SLF001
            SecretString=planted,
            VersionStages=["AWSPENDING"],
        )
        _fail_once(
            secrets_backend, monkeypatch, "put_secret_value", "ResourceExistsException"
        )
        warnings = _spy_warnings(monkeypatch)
        before = await _secret_record(key_id)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert await _secret_record(key_id) == before
        assert await _current_key(secrets_backend, key_id) == key
        assert any(f"{_PREFIX}/{key_id}" in warning for warning in warnings)
        assert not any(planted in warning for warning in warnings)

    async def test_a_container_created_meanwhile_is_written_into(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two instances creating the missing container agree on one.

        Ref: stdapi/tenant_keys.py:_put_pending
        """
        key_id = "a" + "6" * 15
        record: Item = {
            PARTITION_KEY: "TENANT",
            SORT_KEY: f"secret#{key_id}",
            "secret_hash": tenant_keys._hash_secret("o" * 43, b"\0" * 16),  # noqa: SLF001
            "salt": b"\0" * 16,
            "external_id": "legacy",
            "minted_at": 1,
        }
        await put_item(record)
        await put_item(_tenant_item(key_id, key_generation=1))
        original = secrets_backend.create_secret

        async def _lost_the_race(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
            await original(**kwargs)
            error = _client_error("ResourceExistsException", "CreateSecret")
            raise error

        monkeypatch.setattr(secrets_backend, "create_secret", _lost_the_race)

        await reconcile_tenant_keys()

        new = await _current_key(secrets_backend, key_id)
        assert (await verify_tenant_key(new)).key_id == key_id

    async def test_an_already_cleared_pending_label_is_absorbed(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clearing AWSPENDING twice answers InvalidParameterException; it is done.

        Ref: https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_UpdateSecretVersionStage.html
        """
        key_id = "a" + "1" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        original = secrets_backend.update_secret_version_stage

        async def _cleared_already(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
            if kwargs.get("VersionStage") == "AWSPENDING":
                error = _client_error(
                    "InvalidParameterException", "UpdateSecretVersionStage"
                )
                raise error
            return await original(**kwargs)

        monkeypatch.setattr(
            secrets_backend, "update_secret_version_stage", _cleared_already
        )
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        record = await _secret_record(key_id)
        assert "pending_version_id" not in record
        assert record["rotations"] == 1

    async def test_a_promotion_the_service_cannot_confirm_is_left_pending(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The label not landing on the new version keeps the rotation open.

        The record keeps ``pending_version_id`` so the next pass retries the
        promotion, and the operator is told which secret is stuck.

        Ref: stdapi/tenant_keys.py:_promote
        """
        key_id = "a" + "2" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        warnings = _spy_warnings(monkeypatch)
        original = secrets_backend.describe_secret
        current = (await original(SecretId=f"{_PREFIX}/{key_id}"))["VersionIdsToStages"]

        async def _stale(**kwargs: Any) -> Any:  # noqa: ANN401 - the client is untyped
            response = await original(**kwargs)
            return {**response, "VersionIdsToStages": current}

        monkeypatch.setattr(secrets_backend, "describe_secret", _stale)
        monkeypatch.setattr(tenant_keys, "_READ_LAG_SECONDS", 0.0)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        record = await _secret_record(key_id)
        assert record["pending_version_id"] == tenant_keys._version_token(key_id, 1)  # noqa: SLF001
        assert any(f"{_PREFIX}/{key_id}" in warning for warning in warnings)
        monkeypatch.setattr(secrets_backend, "describe_secret", original)
        await reconcile_tenant_keys()
        assert "pending_version_id" not in await _secret_record(key_id)
        assert await _current_key(secrets_backend, key_id) != old

    async def test_a_lost_record_race_leaves_the_promotion_to_the_winner(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two instances rotating one key write one record and promote once.

        Ref: stdapi/tenant_keys.py:_rotate
        """
        key_id = "a" + "3" * 15
        await _declare_and_mint(secrets_backend, key_id=key_id)
        await put_item(_tenant_item(key_id, key_generation=1))
        moves = _record_calls(
            secrets_backend, monkeypatch, "update_secret_version_stage"
        )
        raced = False

        async def _lose_once(item: Item, **kwargs: Any) -> bool:  # noqa: ANN401 - forwarded
            nonlocal raced
            if "pending_version_id" in item and not raced:
                raced = True
                # The other instance's identical write landed first.
                await put_item(item, **kwargs)
                return False
            return await put_item(item, **kwargs)

        monkeypatch.setattr(tenant_keys, "put_item", _lose_once)

        await reconcile_tenant_keys()

        assert raced
        assert not moves, "the loser leaves the promotion to the winner"
        monkeypatch.setattr(tenant_keys, "put_item", put_item)
        await reconcile_tenant_keys()
        assert (await _secret_record(key_id))["rotations"] == 1
        assert "pending_version_id" not in await _secret_record(key_id)

    async def test_a_failed_rotation_is_reported_and_never_raised(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused pending write leaves the key as it was, loudly.

        Ref: stdapi/tenant_keys.py:_rotate_due
             stdapi/api_errors.py:iam_denial_detail
        """
        key_id = "a" + "4" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        warnings = _spy_warnings(monkeypatch)

        async def _denied(**_kwargs: object) -> None:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": (
                            "User: arn:aws:sts::123456789012:assumed-role/x/y is "
                            "not authorized to perform: secretsmanager:PutSecretValue "
                            "on resource: arn:aws:secretsmanager:eu-west-3:"
                            f"123456789012:secret:{_PREFIX}/{key_id}-AbCdEf"
                        ),
                    }
                },
                "PutSecretValue",
            )

        monkeypatch.setattr(secrets_backend, "put_secret_value", _denied)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert await _current_key(secrets_backend, key_id) == key
        assert (await verify_tenant_key(key)).key_id == key_id
        assert any("secretsmanager:PutSecretValue" in w for w in warnings)


class TestOverlap:
    """Both keys authenticate for the overlap, and only for the overlap.

    Ref: stdapi/tenant_keys.py:verify_tenant_key
    """

    async def test_both_keys_authenticate_during_the_overlap(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A client still on the previous key is not locked out by the rotation.

        Ref: stdapi/tenant_keys.py:verify_tenant_key
        """
        key_id = "o" + "1" * 15
        old, new = await _rotate_once(secrets_backend, key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001

        assert (await verify_tenant_key(new)).key_id == key_id
        assert (await verify_tenant_key(old)).key_id == key_id

        record = await _secret_record(key_id)
        until, rotated = record["previous_until"], record["rotated_at"]
        overlap = SETTINGS.tenant_key_rotation_overlap_seconds
        assert isinstance(until, int)
        assert isinstance(rotated, int)
        # The window is the configured one, not merely non-zero; it runs from
        # the promotion, a moment after the record write this compares to.
        assert rotated + overlap <= until <= rotated + overlap + 60

    async def test_the_previous_key_stops_the_instant_the_overlap_ends(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cutoff is checked at verify time, not at the next cache refresh.

        Ref: stdapi/tenant_keys.py:verify_tenant_key
        """
        key_id = "o" + "2" * 15
        old, new = await _rotate_once(secrets_backend, key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        await verify_tenant_key(old)
        until = (await _secret_record(key_id))["previous_until"]
        assert isinstance(until, int)
        reads = _spy_reads(monkeypatch)
        monkeypatch.setattr(tenant_keys, "time", lambda: float(until))

        with pytest.raises(ApiError) as refused:
            await verify_tenant_key(old)

        assert refused.value.status == 401
        assert (await verify_tenant_key(new)).key_id == key_id
        assert reads == [], "the refusal came from the warm cache, without a read"

    async def test_a_zero_overlap_is_a_hard_cutover(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no overlap the previous key is refused as soon as the new one lands.

        Ref: stdapi/tenant_keys.py:_rotate
        """
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", 0)
        key_id = "o" + "3" * 15
        old, new = await _rotate_once(secrets_backend, key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001

        with pytest.raises(ApiError) as refused:
            await verify_tenant_key(old)

        assert refused.value.status == 401
        assert (await verify_tenant_key(new)).key_id == key_id
        assert "previous_secret_hash" not in await _secret_record(key_id)

    async def test_a_zero_overlap_still_covers_the_promotion_window(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hard cutover happens at the promotion, not at the record write.

        Between the two the secret still serves the superseded key, so a
        client re-reading it there would otherwise be handed a key the
        gateway refuses -- the window the promotion's describe, move and
        read-back spans.

        Ref: stdapi/tenant_keys.py:_rotate
        """
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", 0)
        key_id = "o" + "7" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        promote = tenant_keys._promote  # noqa: SLF001
        served: list[str] = []
        accepted: list[str] = []

        async def _check_the_window(*args: Any, **kwargs: Any) -> bool:  # noqa: ANN401 - forwarded
            tenant_keys._CACHE.clear()  # noqa: SLF001
            served.append(await _current_key(secrets_backend, key_id))
            with suppress(ApiError):
                accepted.append((await verify_tenant_key(old)).key_id)
            return await promote(*args, **kwargs)

        monkeypatch.setattr(tenant_keys, "_promote", _check_the_window)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert served == [old], "AWSCURRENT still served the superseded key"
        assert accepted == [key_id], "the key AWSCURRENT served was refused"
        tenant_keys._CACHE.clear()  # noqa: SLF001
        with pytest.raises(ApiError) as refused:
            await verify_tenant_key(old)
        assert refused.value.status == 401
        assert "previous_secret_hash" not in await _secret_record(key_id)

    async def test_the_superseded_key_outlives_a_promotion_that_never_lands(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A promotion the role cannot perform must not lock the tenant out.

        ``secretsmanager:UpdateSecretVersionStage`` missing from the task role
        is first exercised at the first rotation, long after deployment: the
        record stays pending, AWSCURRENT keeps serving the superseded key, and
        that key has to keep authenticating however long the overlap that was
        configured -- otherwise the only accepted key sits in AWSPENDING,
        where no tenant is documented to look.

        Ref: stdapi/tenant_keys.py:_rotate
             docs/operations_authentication_security.md
        """
        overlap = 60
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", overlap)
        key_id = "o" + "8" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        clock = _frozen_clock(monkeypatch)

        _deny_stage_moves(secrets_backend, monkeypatch)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        assert "pending_version_id" in await _secret_record(key_id)
        assert await _current_key(secrets_backend, key_id) == old
        clock[0] += 10 * overlap
        tenant_keys._CACHE.clear()  # noqa: SLF001
        assert (await verify_tenant_key(old)).key_id == key_id

    async def test_the_overlap_runs_from_the_promotion(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once AWSCURRENT has moved the window starts, and ends one overlap later.

        Ref: stdapi/tenant_keys.py:_rotate
        """
        overlap = 60
        stalled = 300
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", overlap)
        key_id = "o" + "9" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        clock = _frozen_clock(monkeypatch)
        promote = tenant_keys._promote  # noqa: SLF001

        async def _stalls(*args: Any, **kwargs: Any) -> bool:  # noqa: ANN401 - forwarded
            clock[0] += stalled
            return await promote(*args, **kwargs)

        monkeypatch.setattr(tenant_keys, "_promote", _stalls)
        await put_item(_tenant_item(key_id, key_generation=1))

        await reconcile_tenant_keys()

        record = await _secret_record(key_id)
        assert "pending_version_id" not in record
        until, rotated = record["previous_until"], record["rotated_at"]
        assert isinstance(until, int)
        assert isinstance(rotated, int)
        assert until == rotated + stalled + overlap
        tenant_keys._CACHE.clear()  # noqa: SLF001
        assert (await verify_tenant_key(old)).key_id == key_id
        clock[0] = float(until)
        with pytest.raises(ApiError) as refused:
            await verify_tenant_key(old)
        assert refused.value.status == 401

    async def test_disabled_wins_over_both_keys(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A suspended tenant is refused whichever of its keys it presents.

        Ref: stdapi/tenant_keys.py:verify_tenant_key
        """
        key_id = "o" + "4" * 15
        old, new = await _rotate_once(secrets_backend, key_id)
        await put_item(_tenant_item(key_id, key_generation=1, disabled=True))
        tenant_keys._CACHE.clear()  # noqa: SLF001

        for key in (old, new):
            with pytest.raises(ApiError) as refused:
                await verify_tenant_key(key)
            assert refused.value.status == 401

    async def test_disabled_wins_over_a_pending_rotation(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suspending a tenant refuses both keys even mid-rotation.

        Ref: stdapi/tenant_keys.py:verify_tenant_key
        """
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", 0)
        key_id = "p" + "1" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)

        _deny_stage_moves(secrets_backend, monkeypatch)
        await put_item(_tenant_item(key_id, key_generation=1))
        await reconcile_tenant_keys()
        assert "pending_version_id" in await _secret_record(key_id)
        await put_item(_tenant_item(key_id, key_generation=1, disabled=True))
        tenant_keys._CACHE.clear()  # noqa: SLF001

        for key in (old, await _pending_key(secrets_backend, key_id)):
            with pytest.raises(ApiError) as refused:
                await verify_tenant_key(key)
            assert refused.value.status == 401

    async def test_a_record_with_a_malformed_overlap_is_refused_as_unavailable(
        self, secrets_backend: SecretsManagerClient
    ) -> None:
        """A previous-secret triple this build cannot read is an operator matter.

        Ref: stdapi/tenant_keys.py:_build_entry
        """
        from stdapi.api_errors import FeatureUnavailableError  # noqa: PLC0415

        key_id = "o" + "6" * 15
        _, new = await _rotate_once(secrets_backend, key_id)
        record = await _secret_record(key_id)
        await put_item({**record, "previous_until": "soon"})
        tenant_keys._CACHE.clear()  # noqa: SLF001

        with pytest.raises(FeatureUnavailableError) as refused:
            await verify_tenant_key(new)

        assert refused.value.status == 503

    async def test_the_previous_key_is_scrubbed_once_the_overlap_ends(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loop drops the superseded hash from the record after the window.

        Ref: stdapi/tenant_keys.py:_scrub_previous
        """
        key_id = "o" + "5" * 15
        await _rotate_once(secrets_backend, key_id)
        until = (await _secret_record(key_id))["previous_until"]
        assert isinstance(until, int)
        monkeypatch.setattr(tenant_keys, "time", lambda: float(until + 1))

        await reconcile_tenant_keys()

        record = await _secret_record(key_id)
        assert "previous_secret_hash" not in record
        assert "previous_salt" not in record
        assert "previous_until" not in record
        assert record["rotations"] == 1


class TestConstantTime:
    """Every verification compares two digests, whatever the record holds.

    Ref: stdapi/tenant_keys.py:_matches
         tests/test_tenant_keys.py:TestVerification
    """

    async def test_a_rotated_key_compares_current_and_previous(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both digests are compared even when the first one already matched.

        Ref: stdapi/tenant_keys.py:_matches
        """
        key_id = "c" + "1" * 15
        old, new = await _rotate_once(secrets_backend, key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        compared = _spy_compares(monkeypatch)

        await verify_tenant_key(new)
        await verify_tenant_key(old)

        assert len(compared) == 4
        assert all(len(left) == len(right) for left, right in compared)

    async def test_a_never_rotated_key_compares_a_dummy_previous(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key with no previous secret still costs two comparisons.

        Ref: stdapi/tenant_keys.py:_matches
        """
        key = await _declare_and_mint(secrets_backend, key_id="c" + "2" * 15)
        compared = _spy_compares(monkeypatch)

        await verify_tenant_key(key)

        assert len(compared) == 2

    async def test_an_unknown_key_id_compares_two_dummies(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fabricated key ID does the same work as a real one, cold and cached.

        Ref: stdapi/tenant_keys.py:_reject_unknown
        """
        del secrets_backend
        compared = _spy_compares(monkeypatch)
        credential = f"{KEY_PREFIX}{_UNKNOWN_KEY_ID}-{_WELL_FORMED_SECRET}"

        with pytest.raises(ApiError):
            await verify_tenant_key(credential)
        cold = len(compared)
        with pytest.raises(ApiError):
            await verify_tenant_key(credential)

        assert cold == 2
        assert len(compared) == 4

    async def test_a_wrong_secret_compares_both_before_refusing(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No short-circuit on the first mismatch.

        Ref: stdapi/tenant_keys.py:_matches
        """
        key_id = "c" + "3" * 15
        _, new = await _rotate_once(secrets_backend, key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        await verify_tenant_key(new)
        monkeypatch.setattr(tenant_keys, "_REFRESH_SECONDS", 0.0)
        compared = _spy_compares(monkeypatch)
        wrong = new[:-4] + ("AAAA" if not new.endswith("AAAA") else "BBBB")

        with pytest.raises(ApiError):
            await verify_tenant_key(wrong)

        # Two on the cached entry, two more on the refreshed one.
        assert len(compared) == 4


class TestRefreshOnMiss:
    """A mismatch against a cached entry re-reads the record, at most once a second.

    Ref: stdapi/tenant_keys.py:verify_tenant_key
    """

    async def test_a_new_key_works_on_a_stale_instance_after_one_read(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client that re-read the secret promptly is not refused for a minute.

        Ref: stdapi/tenant_keys.py:verify_tenant_key
        """
        key_id = "f" + "1" * 15
        old = await _declare_and_mint(secrets_backend, key_id=key_id)
        await verify_tenant_key(old)
        monkeypatch.setattr(tenant_keys, "_REFRESH_SECONDS", 0.0)
        await put_item(_tenant_item(key_id, key_generation=1))
        await reconcile_tenant_keys()
        new = await _current_key(secrets_backend, key_id)
        reads = _spy_reads(monkeypatch)

        assert (await verify_tenant_key(new)).key_id == key_id

        assert reads == [f"tenant#{key_id}", f"secret#{key_id}"]

    async def test_a_flood_of_wrong_secrets_costs_one_read_per_second(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refresh is rate-limited per key, so a wrong secret is not a read amplifier.

        Ref: stdapi/tenant_keys.py:_REFRESH_SECONDS
        """
        key_id = "f" + "2" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        await verify_tenant_key(key)
        clock = [1000.0]
        monkeypatch.setattr(tenant_keys, "monotonic", lambda: clock[0])
        tenant_keys._CACHE.clear()  # noqa: SLF001
        await verify_tenant_key(key)
        reads = _spy_reads(monkeypatch)
        wrong = key[:-4] + ("AAAA" if not key.endswith("AAAA") else "BBBB")

        clock[0] += 1.5
        for _ in range(20):
            with pytest.raises(ApiError):
                await verify_tenant_key(wrong)

        assert len(reads) == 2, "one refresh for the whole burst"
        clock[0] += 1.5
        with pytest.raises(ApiError):
            await verify_tenant_key(wrong)
        assert len(reads) == 4, "a second later, one more"

    async def test_a_fresh_entry_is_not_re_read(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entry read less than a second ago is trusted as it is.

        Ref: stdapi/tenant_keys.py:_REFRESH_SECONDS
        """
        key_id = "f" + "3" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        reads = _spy_reads(monkeypatch)
        wrong = key[:-4] + ("AAAA" if not key.endswith("AAAA") else "BBBB")

        with pytest.raises(ApiError):
            await verify_tenant_key(wrong)

        assert len(reads) == 2, "the cold read only"

    async def test_a_deleted_tenant_is_refused_by_the_refresh(
        self, secrets_backend: SecretsManagerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refresh finding no record is the ordinary unknown-key refusal.

        Ref: stdapi/tenant_keys.py:_lookup
        """
        key_id = "f" + "4" * 15
        key = await _declare_and_mint(secrets_backend, key_id=key_id)
        await verify_tenant_key(key)
        monkeypatch.setattr(tenant_keys, "_REFRESH_SECONDS", 0.0)
        await delete_item("TENANT", f"tenant#{key_id}")
        wrong = key[:-4] + ("AAAA" if not key.endswith("AAAA") else "BBBB")

        with pytest.raises(ApiError) as refused:
            await verify_tenant_key(wrong)

        assert refused.value.status == 401
        assert key_id in tenant_keys._NEGATIVE  # noqa: SLF001


@pytest.mark.gateway("AWS Secrets Manager has no upstream-vendor equivalent")
@pytest.mark.xdist_group("dynamodb")
class TestRealBackends:
    """The label mechanics the stand-in gets differently, against AWS itself.

    The stand-in answers a stage-less read with the version written last and
    never lags a write; the real service resolves ``AWSCURRENT`` and may take
    a moment to show a moved label. Only a real rotation proves the gateway's
    own sequence -- pending write, record, promotion, pending cleared -- ends
    with the tenant reading its new key as current and its old one as
    previous.

    These call the module in process, so they take ``sandbox_dynamodb``: it
    binds the table access to a client opened on the loop the test runs on,
    which the app's own pool cannot be.

    Ref: tests/test_tenant_keys.py:TestRealBackends
    """

    async def test_a_rotation_against_the_real_service(
        self, sandbox_dynamodb: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mint, then rotate, one key through the real service end to end.

        Ref: stdapi/tenant_keys.py:_rotate
             https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_UpdateSecretVersionStage.html
        """
        from asyncio import sleep  # noqa: PLC0415
        from contextlib import suppress  # noqa: PLC0415
        from secrets import token_hex  # noqa: PLC0415

        from stdapi.aws import _CLIENTS  # noqa: PLC0415
        from stdapi.config import AWS_REGION  # noqa: PLC0415

        del sandbox_dynamodb
        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_prefix", _PREFIX)
        monkeypatch.setattr(SETTINGS, "tenant_key_secretsmanager_kms_key_id", None)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_days", None)
        monkeypatch.setattr(SETTINGS, "tenant_key_rotation_overlap_seconds", 604800)
        tenant_keys._CACHE.clear()  # noqa: SLF001
        tenant_keys._NEGATIVE.clear()  # noqa: SLF001
        key_id = token_hex(8)
        secret_name = f"{_PREFIX}/{key_id}"
        session = get_session()
        async with session.create_client(
            "secretsmanager", region_name=AWS_REGION
        ) as client:
            monkeypatch.setitem(_CLIENTS, "secretsmanager", {AWS_REGION: client})
            try:
                minted = await _declare_and_mint(client, key_id=key_id)
                assert (await verify_tenant_key(minted)).key_id == key_id

                await put_item(_tenant_item(key_id, key_generation=1))
                await reconcile_tenant_keys()

                record = await _secret_record(key_id)
                assert record["rotations"] == 1
                assert "pending_version_id" not in record
                # The service's reads lag the promotion by up to a second.
                rotated, previous = minted, None
                for _ in range(20):
                    rotated = await _current_key(client, key_id)
                    with suppress(ClientError):
                        previous = (
                            await client.get_secret_value(
                                SecretId=secret_name, VersionStage="AWSPREVIOUS"
                            )
                        )["SecretString"]
                    if rotated != minted and previous is not None:
                        break
                    await sleep(0.5)
                assert rotated != minted, "the new version never became current"
                assert previous == minted, "the old version never became previous"
                tenant_keys._CACHE.clear()  # noqa: SLF001
                assert (await verify_tenant_key(rotated)).key_id == key_id
                assert (await verify_tenant_key(minted)).key_id == key_id
            finally:
                with suppress(ClientError):
                    await client.delete_secret(
                        SecretId=secret_name, ForceDeleteWithoutRecovery=True
                    )
                await delete_item("TENANT", f"tenant#{key_id}")
                await delete_item("TENANT", f"secret#{key_id}")
