"""Per-tenant API keys, validated against Amazon DynamoDB records.

A tenant key is ``sk-std-<key id>-<secret>``: the key ID is a public
identifier embedded in the token so validation is a direct read, and the
secret is 256 bits of machine entropy the table only ever holds a salted
BLAKE2b-256 hash of. The hash is compared in constant time; a slow KDF would
add tens of milliseconds of CPU to every request to protect a secret that is
already unguessable, and would hand an attacker a CPU-exhaustion lever.

The record is split in two so the operator's declarative tooling and the
server never write the same item:

- ``pk=TENANT``, ``sk=tenant#<key id>`` -- the operator-declared tenant:
  name, ``disabled``, and the scope patterns. Rewritten freely by tooling
  such as the Terraform module.
- ``pk=TENANT``, ``sk=secret#<key id>`` -- the server-minted credential:
  ``secret_hash``, ``salt`` and the ``external_id`` a cross-account role's
  trust policy must require, which the operator reads from here. Never
  written by the operator, so the operator's tooling never sees, stores or
  transports the secret.

Minting closes the gap between the two: a tenant record with no secret record
is pending, and the reconciliation loop mints a secret for it, delivers the
full key to AWS Systems Manager Parameter Store as a ``SecureString`` under
``tenant_key_ssm_parameter_prefix``, and records the salted hash. The
parameter's create-once semantics make the mint idempotent across instances
and crashes: whoever created the parameter defined the secret, and everyone
else derives the hash from it, retrying past the throttle Parameter Store
answers a genuine race with.

Validated keys are cached in-process for ``tenant_key_cache_seconds`` (60 s
by default), which is also the revocation window: a key revoked or edited in
the table keeps its last decision for up to that long on each instance.
Unknown key IDs are negative-cached, bounded in count and time, so a flood of
fabricated keys is neither a read amplifier nor a memory leak. When the
feature is enabled but the table cannot be read, tenant-shaped credentials
are refused with a 503 -- never accepted, and never conflated with a wrong
key -- while every other credential kind is untouched.
"""

from asyncio import CancelledError, Task, create_task, gather, sleep
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from hashlib import blake2b
from hmac import compare_digest
from re import compile as re_compile
from secrets import choice, token_bytes
from time import monotonic, time
from typing import TYPE_CHECKING, Final, NoReturn

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.api_errors import ApiError, FeatureUnavailableError, iam_denial_detail
from stdapi.aws import get_client
from stdapi.aws_dynamodb import (
    PARTITION_KEY,
    SORT_KEY,
    TableUnavailableError,
    delete_item,
    get_item,
    item_key,
    put_item,
    query_partition,
    readable_schema,
)
from stdapi.config import AWS_REGION, SETTINGS
from stdapi.monitoring import (
    EventLog,
    Tenant,
    TenantAwsCredential,
    add_server_warning,
    log_background_event,
    log_error_details,
)
from stdapi.utils import webuuid

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_dynamodb import Item

#: Prefix every tenant API key starts with; dispatchable without a lookup.
KEY_PREFIX: Final = "sk-std-"

#: Length of the public key ID, in base62 characters (~95 bits).
_KEY_ID_LENGTH: Final = 16

#: Length of the secret, in base62 characters (~256 bits of entropy).
_SECRET_LENGTH: Final = 43

#: Alphabet the key ID and secret are drawn from.
_ALPHABET: Final = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

#: Matcher refusing anything but the key alphabet, at either field's length.
_KEY_ID_RE: Final = re_compile(f"^[0-9A-Za-z]{{{_KEY_ID_LENGTH}}}$").match

#: Matcher refusing a secret that is not exactly the minted shape.
_SECRET_RE: Final = re_compile(f"^[0-9A-Za-z]{{{_SECRET_LENGTH}}}$").match

#: Partition every tenant-key record lives in, so one query lists them all.
_PARTITION: Final = "TENANT"

#: Sort-key kind of the operator-declared tenant record.
_TENANT_KIND: Final = "tenant"

#: Sort-key kind of the server-minted credential record.
_SECRET_KIND: Final = "secret"  # noqa: S105 - a record kind, not a credential

#: Size of the stored secret hash, in bytes.
_HASH_SIZE: Final = 32

#: Size of the per-key random salt, in bytes.
_SALT_SIZE: Final = 16

#: Most recently used validated keys kept in the per-process cache.
_CACHE_MAX: Final = 4096

#: Seconds an unknown key ID is refused without a table read.
_NEGATIVE_TTL: Final = 10.0

#: Most recently refused unknown key IDs kept, bounding the memory a flood costs.
_NEGATIVE_MAX: Final = 1024

#: Seconds between reconciliation runs, which is how long a newly declared tenant waits for its key.
_RECONCILE_INTERVAL: Final = 60.0

#: Feature name tenant-key refusals answer with.
_FEATURE: Final = "Tenant API key authentication"

#: Salt hashing the secret of an unknown key ID, evening out the refusal paths.
_DUMMY_SALT: Final = token_bytes(_SALT_SIZE)

#: Region the minted keys are delivered through.
_SSM_REGION: RegionName = AWS_REGION  # type: ignore[assignment]

#: Parameter Store's answer to concurrent writes of one parameter, which the mint races into.
_THROTTLED: Final = "TooManyUpdates"

#: Attempts at the create-once write before the mint is left to the next reconciliation.
_MINT_ATTEMPTS: Final = 5

#: Seconds between those attempts, long enough for the winner's write to land.
_MINT_RETRY_SECONDS: Final = 0.2

#: Matcher a tenant record's cross-account IAM role ARN must satisfy.
_ROLE_ARN_RE: Final = re_compile(
    r"^arn:aws[a-z-]*:iam::\d{12}:role/[\w+=,.@/-]+$"
).match


@dataclass(frozen=True, slots=True)
class _Entry:
    """One tenant key as last read from the table.

    Attributes:
        tenant: The tenant the key belongs to, scopes included.
        disabled: Whether the operator disabled the key.
        secret_hash: Salted BLAKE2b-256 of the secret.
        salt: Salt the hash was computed with.
        fetched: Monotonic clock reading of the table read.
    """

    tenant: Tenant
    disabled: bool
    secret_hash: bytes
    salt: bytes
    fetched: float


#: Validated keys by key ID, most recently stored last.
_CACHE: OrderedDict[str, _Entry] = OrderedDict()

#: Refusal deadline by unknown key ID, most recently stored last.
_NEGATIVE: OrderedDict[str, float] = OrderedDict()

#: Handle of the periodic reconciliation loop, None while not running.
_RECONCILE_TASK: Task[None] | None = None

#: Key IDs whose broken records were already reported, so the loop does not repeat itself.
_REPORTED: set[str] = set()


def tenant_key_client_specs() -> tuple[tuple[str, str | None], ...]:
    """Return the client specs the connection pool needs for key delivery.

    Returns:
        One ``("ssm", region)`` spec when tenant keys are enabled, and nothing
        at all otherwise.
    """
    return (("ssm", _SSM_REGION),) if SETTINGS.tenant_api_keys else ()


def is_tenant_key(credential: str) -> bool:
    """Whether *credential* is shaped like a tenant API key.

    Args:
        credential: The credential the caller presented.

    Returns:
        True when it carries the tenant key prefix.
    """
    return credential.startswith(KEY_PREFIX)


def _parse(credential: str) -> tuple[str, str] | None:
    """Split a tenant-shaped credential into its key ID and secret.

    Args:
        credential: A credential carrying :data:`KEY_PREFIX`.

    Returns:
        The key ID and the secret, or None when the shape is not a minted
        key's -- wrong lengths, wrong alphabet or a missing separator.
    """
    body = credential[len(KEY_PREFIX) :]
    if len(body) != _KEY_ID_LENGTH + 1 + _SECRET_LENGTH:
        return None
    key_id, separator, secret = (
        body[:_KEY_ID_LENGTH],
        body[_KEY_ID_LENGTH],
        body[_KEY_ID_LENGTH + 1 :],
    )
    if separator != "-" or not _KEY_ID_RE(key_id) or not _SECRET_RE(secret):
        return None
    return key_id, secret


def _hash_secret(secret: str, salt: bytes) -> bytes:
    """Hash a key secret with its salt.

    Args:
        secret: The secret, as presented or as minted.
        salt: The per-key random salt.

    Returns:
        The salted BLAKE2b-256 digest.
    """
    return blake2b(secret.encode(), digest_size=_HASH_SIZE, salt=salt).digest()


def _refuse(detail: str) -> NoReturn:
    """Answer the detail-free 401 every refused credential gets.

    Args:
        detail: What was wrong, for the request log only.

    Raises:
        ApiError: Always, carrying nothing a caller could probe with.
    """
    log_error_details(detail)
    msg = "Unauthorized"
    raise ApiError(msg, status=401)


def _malformed_record(key_id: str, what: str) -> FeatureUnavailableError:
    """Refuse a key whose stored record this build cannot trust.

    Args:
        key_id: The key the record belongs to.
        what: What is wrong with the record, named for the operator.

    Returns:
        The error to raise, already logged for the operator.
    """
    return FeatureUnavailableError(
        _FEATURE,
        f"The record of tenant key '{key_id}' in the '{SETTINGS.aws_dynamodb_table}' "
        f"DynamoDB table cannot be used: {what}",
    )


def _patterns(item: Item, attribute: str, key_id: str) -> tuple[str, ...] | None:
    """Read one scope pattern list off a tenant record.

    Args:
        item: The tenant record.
        attribute: The list's attribute name.
        key_id: The key the record belongs to, for the operator's log line.

    Returns:
        The patterns, or None when the operator never set the attribute --
        which restricts nothing, where an empty list allows nothing.

    Raises:
        FeatureUnavailableError: The attribute is not a list of strings.
    """
    value = item.get(attribute)
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(pattern, str) for pattern in value
    ):
        raise _malformed_record(key_id, f"'{attribute}' is not a list of strings")
    return tuple(value)  # type: ignore[arg-type]


def _aws_credential(
    key_id: str, tenant_item: Item, secret_item: Item
) -> TenantAwsCredential | None:
    """Read the cross-account AWS credential off a key's records, if declared.

    Fails closed on every half-configured state: a declared role must never be
    silently ignored, or the tenant's usage lands on the deployment's bill
    while the operator believes it does not.

    Args:
        key_id: The key both records belong to.
        tenant_item: The operator-declared tenant record.
        secret_item: The server-minted credential record.

    Returns:
        The credential, or None when the record declares no role.

    Raises:
        FeatureUnavailableError: The role ARN is malformed, the feature is
            disabled while a role is declared, or the external ID is not
            minted yet.
    """
    role_arn = tenant_item.get("aws_role_arn")
    if role_arn is None:
        return None
    if not isinstance(role_arn, str) or not _ROLE_ARN_RE(role_arn):
        raise _malformed_record(key_id, "'aws_role_arn' is not an IAM role ARN")
    if not SETTINGS.tenant_aws_credentials:
        raise _malformed_record(
            key_id,
            "it declares 'aws_role_arn' while tenant_aws_credentials is "
            "disabled; enable the setting or remove the attribute",
        )
    external_id = secret_item.get("external_id")
    if not isinstance(external_id, str) or not external_id:
        raise _malformed_record(
            key_id,
            "its ExternalId is not minted yet; the server mints one within "
            "a minute of the role being declared",
        )
    return TenantAwsCredential(role_arn=role_arn, external_id=external_id)


def _build_entry(key_id: str, tenant_item: Item, secret_item: Item) -> _Entry:
    """Assemble a cache entry from the two records of one key.

    Args:
        key_id: The key both records belong to.
        tenant_item: The operator-declared tenant record.
        secret_item: The server-minted credential record.

    Returns:
        The entry.

    Raises:
        FeatureUnavailableError: A record was written by a newer build, or an
            attribute does not hold what this build stores there.
    """
    if not readable_schema(tenant_item) or not readable_schema(secret_item):
        raise _malformed_record(key_id, "it was written by a newer server version")
    secret_hash = secret_item.get("secret_hash")
    salt = secret_item.get("salt")
    if not isinstance(secret_hash, bytes) or not isinstance(salt, bytes):
        raise _malformed_record(key_id, "'secret_hash' or 'salt' is not binary")
    name = tenant_item.get("name")
    return _Entry(
        tenant=Tenant(
            key_id=key_id,
            name=name if isinstance(name, str) and name else key_id,
            models_allow=_patterns(tenant_item, "models_allow", key_id),
            models_deny=_patterns(tenant_item, "models_deny", key_id) or (),
            endpoints_allow=_patterns(tenant_item, "endpoints_allow", key_id),
            endpoints_deny=_patterns(tenant_item, "endpoints_deny", key_id) or (),
            aws_credential=_aws_credential(key_id, tenant_item, secret_item),
        ),
        disabled=bool(tenant_item.get("disabled")),
        secret_hash=secret_hash,
        salt=salt,
        fetched=monotonic(),
    )


def _reject_unknown(secret: str) -> NoReturn:
    """Refuse a key ID the table does not hold.

    Args:
        secret: The presented secret, hashed against a fixed salt so this
            refusal costs the same work as a wrong-secret one.

    Raises:
        ApiError: Always, identical to a wrong-secret refusal.
    """
    compare_digest(_hash_secret(secret, _DUMMY_SALT), _DUMMY_SALT + _DUMMY_SALT)
    _refuse("Unknown tenant API key")


async def _lookup(key_id: str, secret: str) -> _Entry:
    """Return the cached or freshly read entry for *key_id*.

    Args:
        key_id: The key to look up.
        secret: The presented secret, for the unknown-key refusal only.

    Raises:
        ApiError: 401 when the table holds no such key.
        FeatureUnavailableError: The table cannot be read, or the record
            cannot be used.

    Returns:
        The entry, no older than ``tenant_key_cache_seconds``.
    """
    now = monotonic()
    entry = _CACHE.get(key_id)
    if entry is not None and now - entry.fetched > SETTINGS.tenant_key_cache_seconds:
        _CACHE.pop(key_id, None)
        entry = None
    if entry is not None:
        return entry
    deadline = _NEGATIVE.get(key_id)
    if deadline is not None:
        if now < deadline:
            _reject_unknown(secret)
        _NEGATIVE.pop(key_id, None)
    try:
        tenant_item, secret_item = await gather(
            get_item(_PARTITION, item_key(_TENANT_KIND, key_id)),
            get_item(_PARTITION, item_key(_SECRET_KIND, key_id)),
        )
    except TableUnavailableError as error:
        raise FeatureUnavailableError(_FEATURE, error.detail) from error
    if tenant_item is None or secret_item is None:
        _NEGATIVE[key_id] = now + _NEGATIVE_TTL
        _NEGATIVE.move_to_end(key_id)
        while len(_NEGATIVE) > _NEGATIVE_MAX:
            _NEGATIVE.popitem(last=False)
        _reject_unknown(secret)
    entry = _build_entry(key_id, tenant_item, secret_item)
    _CACHE[key_id] = entry
    _CACHE.move_to_end(key_id)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return entry


async def verify_tenant_key(credential: str) -> Tenant:
    """Verify a tenant-shaped credential and return its tenant.

    Args:
        credential: The credential the caller presented, carrying
            :data:`KEY_PREFIX`.

    Returns:
        The verified tenant, scopes included.

    Raises:
        ApiError: 401 when the key is malformed, unknown, wrong or disabled.
        FeatureUnavailableError: The table cannot be read, or the record
            cannot be used; a valid key is never refused as unauthorized for
            an operational failure.
    """
    parsed = _parse(credential)
    if parsed is None:
        _refuse("Malformed tenant API key")
    key_id, secret = parsed
    entry = await _lookup(key_id, secret)
    if not compare_digest(_hash_secret(secret, entry.salt), entry.secret_hash):
        _refuse("Invalid tenant API key")
    if entry.disabled:
        _refuse(f"Tenant API key '{key_id}' is disabled")
    return entry.tenant


async def resume_tenant(key_id: str) -> Tenant:
    """Return the tenant behind an already-verified grant, if it still stands.

    A minted Realtime client secret proves a tenant-authenticated request
    happened; what must be re-checked at connect time is that the tenant still
    exists and is not disabled, so revocation reaches sessions opened later.

    Args:
        key_id: The key ID the grant was issued under.

    Returns:
        The tenant, scopes included.

    Raises:
        ApiError: 401 when the key no longer exists or is disabled.
        FeatureUnavailableError: The table cannot be read, or the record
            cannot be used.
    """
    if not _KEY_ID_RE(key_id):
        _refuse("Malformed tenant key ID")
    entry = await _lookup(key_id, "")
    if entry.disabled:
        _refuse(f"Tenant API key '{key_id}' is disabled")
    return entry.tenant


def _delivery_parameter(key_id: str) -> str:
    """Name the Parameter Store parameter one key is delivered through.

    Args:
        key_id: The key being delivered.

    Returns:
        The parameter name.
    """
    return f"{SETTINGS.tenant_key_ssm_parameter_prefix}/{key_id}"


async def _mint(key_id: str, name: str) -> None:
    """Mint, deliver and record the secret of one pending tenant key.

    The Parameter Store write is create-once, which makes the whole mint
    idempotent: the instance that created the parameter defined the secret,
    and any other instance -- or a retry after a crash between delivery and
    recording -- reads the parameter back and records the same secret's hash.

    A genuine race is not answered with that create-once refusal, though:
    Parameter Store throttles concurrent writes of one name with
    :data:`_THROTTLED` before either write lands, so a loser has to retry to
    find the winner's parameter rather than take it for absent.

    Args:
        key_id: The key to mint.
        name: The tenant's declared name, for the parameter description.

    Raises:
        ClientError: The parameter could not be written or read back.
        TableUnavailableError: The credential record could not be written.
    """
    parameter = _delivery_parameter(key_id)
    secret = "".join(choice(_ALPHABET) for _ in range(_SECRET_LENGTH))
    ssm_client = get_client("ssm", _SSM_REGION)
    # Unset leaves KeyId out entirely, which is what selects 'alias/aws/ssm'.
    key_kwargs = (
        {"KeyId": SETTINGS.tenant_key_ssm_kms_key_id}
        if SETTINGS.tenant_key_ssm_kms_key_id
        else {}
    )
    for remaining in range(_MINT_ATTEMPTS - 1, -1, -1):
        try:
            await ssm_client.put_parameter(
                Name=parameter,
                Value=f"{KEY_PREFIX}{key_id}-{secret}",
                Type="SecureString",
                Overwrite=False,
                Description=(
                    f"stdapi.ai API key of tenant '{name}'. "
                    "Deliver it to the tenant, then delete this parameter."
                ),
                **key_kwargs,
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code == _THROTTLED and remaining:
                await sleep(_MINT_RETRY_SECONDS)
                continue
            if code != "ParameterAlreadyExists":
                raise
            value = (
                await ssm_client.get_parameter(Name=parameter, WithDecryption=True)
            )["Parameter"]["Value"]
            recovered = _parse(value) if is_tenant_key(value) else None
            if recovered is None or recovered[0] != key_id:
                log_error_details(
                    f"SSM parameter '{parameter}' does not hold tenant key "
                    f"'{key_id}': delete the parameter to let the server mint one",
                    level="warning",
                )
                return
            secret = recovered[1]
        break
    salt = token_bytes(_SALT_SIZE)
    # Minted with every key so registering a role later needs no write.
    external_id = webuuid()
    written = await put_item(
        {
            PARTITION_KEY: _PARTITION,
            SORT_KEY: item_key(_SECRET_KIND, key_id),
            "secret_hash": _hash_secret(secret, salt),
            "salt": salt,
            "external_id": external_id,
            "minted_at": int(time()),
        },
        condition=f"attribute_not_exists({PARTITION_KEY})",
    )
    log_error_details(
        f"Minted tenant API key '{key_id}' into SSM parameter '{parameter}', "
        "with an ExternalId for a cross-account role in its credential record"
        if written
        else f"Tenant API key '{key_id}' was minted by another instance",
        level="info",
    )


async def reconcile_tenant_keys() -> None:
    """Mint every pending tenant key and drop every orphaned credential record.

    A tenant record with no credential record is pending: the operator's
    tooling declared it and the secret does not exist yet. A credential record
    with no tenant record is orphaned: the tenant was destroyed and the hash
    is inert, so it is removed.

    Raises:
        TableUnavailableError: The partition could not be listed.
    """
    tenants: dict[str, Item] = {}
    secrets: dict[str, Item] = {}
    # Sort key each credential record was read under, so a record whose key ID
    # is out of spec is still addressable without rebuilding its key.
    secret_sort_keys: dict[str, str] = {}
    # Consistent, once a minute: a stale read here could mistake a freshly
    # declared tenant's credential for an orphan and revoke a delivered key.
    for item in await query_partition(_PARTITION, consistent=True):
        sort_key = item.get(SORT_KEY)
        if not isinstance(sort_key, str):
            continue
        kind, _, key_id = sort_key.partition("#")
        if kind == _TENANT_KIND and key_id:
            tenants[key_id] = item
        elif kind == _SECRET_KIND and key_id:
            secrets[key_id] = item
            secret_sort_keys[key_id] = sort_key
    pending = {
        key_id: item for key_id, item in tenants.items() if key_id not in secrets
    }
    orphans = {
        key_id: secret_sort_keys[key_id] for key_id in secrets.keys() - tenants.keys()
    }
    # Credential records minted before ExternalId existed, now needing one.
    unminted_external = {
        key_id: item
        for key_id, item in secrets.items()
        if key_id in tenants
        and tenants[key_id].get("aws_role_arn") is not None
        and not item.get("external_id")
    }
    if not pending and not orphans and not unminted_external:
        return
    with log_background_event("tenant_keys_reconcile", webuuid()):
        for key_id, item in pending.items():
            await _mint_pending(key_id, item)
        for key_id, item in unminted_external.items():
            await _backfill_external_id(key_id, item)
        for key_id, sort_key in orphans.items():
            await _drop_orphan(key_id, sort_key)


async def _drop_orphan(key_id: str, sort_key: str) -> None:
    """Remove the credential record of a tenant that no longer exists.

    Deleting one revokes a credential, so the tenant record is re-read
    immediately before: a tooling-driven destroy-and-recreate would otherwise
    let a pass started inside that gap revoke a key already delivered.

    Args:
        key_id: The key the record belongs to.
        sort_key: The sort key the record was read under, which is what it is
            deleted by -- a key ID no server ever minted has no rebuildable key.

    Raises:
        TableUnavailableError: The record could not be re-read or deleted.
    """
    if not _KEY_ID_RE(key_id):
        if key_id not in _REPORTED:
            _REPORTED.add(key_id)
            log_error_details(
                f"Credential record '{sort_key}' carries a key ID this server "
                "never mints and is left untouched: remove it with the tooling "
                "that wrote it",
                level="warning",
            )
        return
    recreated = await get_item(
        _PARTITION, item_key(_TENANT_KIND, key_id), consistent=True
    )
    if recreated is not None:
        return
    await delete_item(_PARTITION, sort_key)
    log_error_details(
        f"Revoked the credential record of destroyed tenant key '{key_id}'",
        level="warning",
    )


async def _backfill_external_id(key_id: str, secret_item: Item) -> None:
    """Mint the ExternalId of a credential record that predates the feature.

    Create-once: the conditional write makes concurrent instances agree on a
    single value, exactly like the secret mint itself. Failures are reported
    rather than raised; the next reconciliation retries.

    Args:
        key_id: The key whose credential record lacks an ExternalId.
        secret_item: The credential record, as read by the reconciliation.
    """
    external_id = webuuid()
    try:
        written = await put_item(
            {**secret_item, "external_id": external_id},
            condition="attribute_not_exists(external_id)",
        )
    except (ClientError, BotoCoreError, TableUnavailableError) as error:
        detail = (
            error.detail
            if isinstance(error, TableUnavailableError)
            else type(error).__name__
        )
        log_error_details(
            f"The ExternalId of tenant key '{key_id}' could not be minted: {detail}",
            level="warning",
        )
        return
    log_error_details(
        f"Minted the ExternalId of tenant key '{key_id}' into its credential "
        "record: the tenant must require it in its role's trust policy"
        if written
        else f"The ExternalId of tenant key '{key_id}' was minted by another instance",
        level="info",
    )


def _mint_failure_detail(key_id: str, error: ClientError | BotoCoreError) -> str:
    """Describe why delivering one tenant key failed, for the operator log.

    Reuses :func:`stdapi.api_errors.iam_denial_detail`, so a missing permission
    reads the same as every other AWS denial the server reports -- including
    one on the KMS key encrypting the parameter, which fails the same
    ``PutParameter``/``GetParameter`` call and is named by its own action
    rather than guessed at.

    Args:
        key_id: The key that could not be delivered.
        error: The failure raised writing or reading back the parameter.

    Returns:
        Which call failed, on which parameter, and why -- the missing IAM
        permission when AWS denied it, else the bare error code. Never the
        AWS error's own message text, which may name the caller's principal.
    """
    parameter = _delivery_parameter(key_id)
    if not isinstance(error, ClientError):
        return f"delivering to '{parameter}' could not be sent ({type(error).__name__})"
    where = f"{error.operation_name} on '{parameter}'"
    if denial := iam_denial_detail(error):
        return f"{where} was denied: {denial}"
    code = error.response.get("Error", {}).get("Code", "")
    return f"{where} failed ({code})" if code else where


async def _mint_pending(key_id: str, item: Item) -> None:
    """Mint one pending tenant key, reporting rather than raising failures.

    Args:
        key_id: The pending key.
        item: Its tenant record.
    """
    if not _KEY_ID_RE(key_id) or not readable_schema(item):
        if key_id not in _REPORTED:
            _REPORTED.add(key_id)
            log_error_details(
                f"Tenant record '{key_id}' cannot be minted: the key ID "
                "or the record layout is not one this server writes",
                level="warning",
            )
        return
    name = item.get("name")
    try:
        await _mint(key_id, name if isinstance(name, str) else key_id)
    except (ClientError, BotoCoreError, TableUnavailableError) as error:
        detail = (
            error.detail
            if isinstance(error, TableUnavailableError)
            else _mint_failure_detail(key_id, error)
        )
        log_error_details(
            f"Tenant API key '{key_id}' could not be minted: {detail}", level="warning"
        )


async def initialize_tenant_keys(start_event: EventLog) -> None:
    """Run the first reconciliation at startup, when tenant keys are enabled.

    Reported and never fatal: a table or parameter a moment away from existing
    must not turn into an outage, and validation fails closed on its own terms
    until the table is reachable.

    Args:
        start_event: Startup event log any finding is reported on.
    """
    if not SETTINGS.tenant_api_keys:
        return
    key_id = SETTINGS.tenant_key_ssm_kms_key_id
    # Parameter Store is regional and cannot encrypt with a key from elsewhere.
    # Warned rather than refused, and checked here rather than beside the shape
    # validations in config.py: the region is only detected once SETTINGS exists,
    # and refusing would take a whole deployment down over one optional delivery
    # setting. A mint failure names the KMS permission the call was denied.
    if key_id and key_id.startswith("arn:") and key_id.split(":")[3] != _SSM_REGION:
        add_server_warning(
            start_event,
            f"Tenant API keys cannot be delivered: their KMS key '{key_id}' is "
            f"not in region '{_SSM_REGION}'",
        )
    try:
        await reconcile_tenant_keys()
    except (TableUnavailableError, ClientError, BotoCoreError) as error:
        detail = (
            error.detail
            if isinstance(error, TableUnavailableError)
            else type(error).__name__
        )
        add_server_warning(
            start_event, f"Tenant API keys cannot be reconciled yet: {detail}"
        )


def open_tenant_key_reconciliation() -> None:
    """Start the periodic reconciliation loop, when tenant keys are enabled."""
    global _RECONCILE_TASK  # noqa: PLW0603
    if SETTINGS.tenant_api_keys and _RECONCILE_TASK is None:
        _RECONCILE_TASK = create_task(_reconcile_loop())


async def close_tenant_key_reconciliation() -> None:
    """Stop the periodic reconciliation loop, if it is running."""
    global _RECONCILE_TASK  # noqa: PLW0603
    if (task := _RECONCILE_TASK) is not None:
        _RECONCILE_TASK = None
        task.cancel()
        with suppress(CancelledError):
            await task


async def _reconcile_loop() -> None:
    """Reconcile forever, reporting failures without ever stopping."""
    while True:
        await sleep(_RECONCILE_INTERVAL)
        try:
            await reconcile_tenant_keys()
        except (TableUnavailableError, ClientError, BotoCoreError) as error:
            with log_background_event("tenant_keys_reconcile", webuuid()):
                log_error_details(
                    error.detail
                    if isinstance(error, TableUnavailableError)
                    else f"Tenant keys could not be reconciled: {type(error).__name__}",
                    level="warning",
                )
