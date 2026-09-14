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
  name, ``disabled``, the scope patterns and ``key_generation``. Rewritten
  freely by tooling such as the Terraform module.
- ``pk=TENANT``, ``sk=secret#<key id>`` -- the server-minted credential:
  ``secret_hash``, ``salt``, the ``external_id`` a cross-account role's trust
  policy must require, and the rotation state. Never written by the
  operator, so the operator's tooling never sees, stores or transports the
  secret.

Minting closes the gap between the two: a tenant record with no secret record
is pending, and the reconciliation loop mints a secret for it, publishes the
full key through one of two stores, and records the salted hash. By default
the key is delivered once as an AWS Systems Manager Parameter Store
``SecureString`` under ``tenant_key_ssm_parameter_prefix``, which the
operator reads and deletes. With ``tenant_key_secretsmanager_prefix`` set it
is stored durably as the current version of an AWS Secrets Manager secret
instead, which is what makes rotation possible: the loop writes a new key as
the secret's ``AWSPENDING`` version, records its hash, promotes it to
``AWSCURRENT`` -- the superseded key becoming ``AWSPREVIOUS`` -- and keeps
accepting the superseded key for ``tenant_key_rotation_overlap_seconds``
counted from that promotion, so a client that re-reads the secret on its own
schedule is never locked out.
Every store write is create-once or carries a deterministic request token,
which makes minting and every rotation step idempotent across instances and
crashes: whoever wrote the version defined the secret, and everyone else
reads it back and records the same hash.

Validated keys are cached in-process for ``tenant_key_cache_seconds`` (60 s
by default), which is also the revocation window: a key revoked or edited in
the table keeps its last decision for up to that long on each instance. A
key that does not match its cached entry is re-read at most once a second,
so a freshly rotated key works everywhere within one read. Unknown key IDs
are negative-cached, bounded in count and time, so a flood of fabricated
keys is neither a read amplifier nor a memory leak. When the feature is
enabled but the table cannot be read, tenant-shaped credentials are refused
with a 503 -- never accepted, and never conflated with a wrong key -- while
every other credential kind is untouched.
"""

from asyncio import CancelledError, Task, create_task, gather, sleep
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from hashlib import blake2b
from hmac import compare_digest
from math import inf
from re import compile as re_compile
from secrets import choice, token_bytes
from time import monotonic, time
from typing import TYPE_CHECKING, Any, Final, NoReturn
from uuid import UUID, uuid5

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.api_errors import FeatureUnavailableError, iam_denial_detail, unauthorized
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

#: Matcher refusing anything but the key alphabet, at the key ID's length.
_KEY_ID_RE: Final = re_compile(f"[0-9A-Za-z]{{{_KEY_ID_LENGTH}}}").fullmatch

#: Matcher splitting a full credential into its key ID and secret.
_KEY_RE: Final = re_compile(
    rf"{KEY_PREFIX}([0-9A-Za-z]{{{_KEY_ID_LENGTH}}})-([0-9A-Za-z]{{{_SECRET_LENGTH}}})"
).fullmatch

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

#: Seconds a cached entry must be old before a mismatching secret re-reads it.
_REFRESH_SECONDS: Final = 1.0

#: Seconds between reconciliation runs, which is how long a newly declared tenant waits for its key.
_RECONCILE_INTERVAL: Final = 60.0

#: Feature name tenant-key refusals answer with.
_FEATURE: Final = "Tenant API key authentication"

#: Salt hashing the secret of an unknown key ID, evening out the refusal paths.
_DUMMY_SALT: Final = token_bytes(_SALT_SIZE)

#: Digest standing in for a previous secret no record holds; never matches anything.
_DUMMY_HASH: Final = token_bytes(_HASH_SIZE)

#: Region the key store lives in, whichever store is configured.
_STORE_REGION: RegionName = AWS_REGION  # type: ignore[assignment]

#: Parameter Store's answer to concurrent writes of one parameter, which the mint races into.
_THROTTLED: Final = "TooManyUpdates"

#: Attempts at the create-once write before the mint is left to the next reconciliation.
_MINT_ATTEMPTS: Final = 5

#: Seconds between those attempts, long enough for the winner's write to land.
_MINT_RETRY_SECONDS: Final = 0.2

#: Attempts at reading a secret version just written, which Secrets Manager reads lag.
_READ_LAG_ATTEMPTS: Final = 5

#: Seconds between those attempts; a promotion took 0.7 s to become readable when measured.
_READ_LAG_SECONDS: Final = 0.5

#: Staging label of the version a tenant reads as its key.
_CURRENT_STAGE: Final = "AWSCURRENT"

#: Staging label a rotation writes the new key under before promoting it.
_PENDING_STAGE: Final = "AWSPENDING"

#: Namespace deriving the deterministic request token of each secret version.
_TOKEN_NAMESPACE: Final = UUID("6f1d0c1e-3c4b-4a8e-9a5c-2e6b8d1f4a70")

#: Seconds in a day, the unit of the rotation schedule.
_DAY_SECONDS: Final = 86400

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
        previous_secret_hash: Digest of the secret a rotation superseded, or
            a dummy that matches nothing.
        previous_salt: Salt of that digest.
        previous_until: Epoch second the superseded secret stops
            authenticating at; 0 when there is none, and unbounded while the
            rotation that superseded it has not promoted its replacement.
        fetched: Monotonic clock reading of the table read.
    """

    tenant: Tenant
    disabled: bool
    secret_hash: bytes
    salt: bytes
    previous_secret_hash: bytes
    previous_salt: bytes
    previous_until: float
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
    """Return the client specs the connection pool needs for the key store.

    Returns:
        One ``("secretsmanager", region)`` spec when keys are stored in AWS
        Secrets Manager, one ``("ssm", region)`` spec when they are delivered
        through Parameter Store, and nothing at all while tenant keys are off.
    """
    if not SETTINGS.tenant_api_keys:
        return ()
    service = "secretsmanager" if SETTINGS.tenant_key_secretsmanager_prefix else "ssm"
    return ((service, _STORE_REGION),)


def is_tenant_key(credential: str) -> bool:
    """Whether *credential* is shaped like a tenant API key.

    Args:
        credential: The credential the caller presented.

    Returns:
        True when it carries the tenant key prefix.
    """
    return credential.startswith(KEY_PREFIX)


def _parse(credential: str) -> tuple[str, str] | None:
    """Split a credential into its key ID and secret.

    Args:
        credential: The credential the caller presented.

    Returns:
        The key ID and the secret, or None when the shape is not a minted
        key's -- missing prefix, wrong lengths or wrong alphabet.
    """
    return (m[1], m[2]) if (m := _KEY_RE(credential)) else None


def _hash_secret(secret: str, salt: bytes) -> bytes:
    """Hash a key secret with its salt.

    Args:
        secret: The secret, as presented or as minted.
        salt: The per-key random salt.

    Returns:
        The salted BLAKE2b-256 digest.
    """
    return blake2b(secret.encode(), digest_size=_HASH_SIZE, salt=salt).digest()


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


def _digest_pair(
    key_id: str, secret_item: Item, hash_attribute: str, salt_attribute: str
) -> tuple[bytes, bytes]:
    """Read one hash-and-salt pair off a credential record.

    Args:
        key_id: The key the record belongs to.
        secret_item: The server-minted credential record.
        hash_attribute: The digest's attribute name.
        salt_attribute: The salt's attribute name.

    Returns:
        The digest and the salt.

    Raises:
        FeatureUnavailableError: Either is not binary, or not the size this
            build writes -- an off-size salt is refused rather than hashed
            with, since BLAKE2b rejects one longer than its salt parameter.
    """
    secret_hash = secret_item.get(hash_attribute)
    salt = secret_item.get(salt_attribute)
    if not isinstance(secret_hash, bytes) or not isinstance(salt, bytes):
        raise _malformed_record(
            key_id, f"'{hash_attribute}' or '{salt_attribute}' is not binary"
        )
    if len(secret_hash) != _HASH_SIZE or len(salt) != _SALT_SIZE:
        raise _malformed_record(
            key_id,
            f"'{hash_attribute}' or '{salt_attribute}' does not have the size "
            "this build writes",
        )
    return secret_hash, salt


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
    secret_hash, salt = _digest_pair(key_id, secret_item, "secret_hash", "salt")
    previous_until = secret_item.get("previous_until")
    if previous_until is None:
        previous_hash, previous_salt = _DUMMY_HASH, _DUMMY_SALT
        deadline = 0.0
    else:
        previous_hash, previous_salt = _digest_pair(
            key_id, secret_item, "previous_secret_hash", "previous_salt"
        )
        if not isinstance(previous_until, int):
            raise _malformed_record(key_id, "'previous_until' is not a number")
        # A pending version means AWSCURRENT still serves the superseded
        # secret: the overlap only starts once the promotion has landed.
        deadline = (
            inf
            if secret_item.get("pending_version_id") is not None
            else float(previous_until)
        )
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
        previous_secret_hash=previous_hash,
        previous_salt=previous_salt,
        previous_until=deadline,
        fetched=monotonic(),
    )


def _matches(secret: str, entry: _Entry) -> bool:
    """Whether a presented secret is the entry's current or still-valid previous one.

    Both digests are always computed and compared, combined without a
    short-circuit, so a refusal costs the same work whatever the record
    holds and whichever of the two would have matched.

    Args:
        secret: The presented secret.
        entry: The key's entry.

    Returns:
        True when the secret authenticates.
    """
    current = compare_digest(_hash_secret(secret, entry.salt), entry.secret_hash)
    previous = compare_digest(
        _hash_secret(secret, entry.previous_salt), entry.previous_secret_hash
    )
    return current | (previous & (time() < entry.previous_until))


def _reject_unknown(secret: str) -> NoReturn:
    """Refuse a key ID the table does not hold.

    Args:
        secret: The presented secret, hashed and compared twice against fixed
            values so this refusal costs the same work as a wrong-secret one.

    Raises:
        ApiError: Always, identical to a wrong-secret refusal.
    """
    compare_digest(_hash_secret(secret, _DUMMY_SALT), _DUMMY_HASH)
    compare_digest(_hash_secret(secret, _DUMMY_SALT), _DUMMY_HASH)
    unauthorized("Unknown tenant API key")


async def _lookup(key_id: str, secret: str, *, refresh: bool = False) -> _Entry:
    """Return the cached or freshly read entry for *key_id*.

    Args:
        key_id: The key to look up.
        secret: The presented secret, for the unknown-key refusal only.
        refresh: Read the table even when a fresh entry is cached.

    Raises:
        ApiError: 401 when the table holds no such key.
        FeatureUnavailableError: The table cannot be read, or the record
            cannot be used.

    Returns:
        The entry, no older than ``tenant_key_cache_seconds``.
    """
    now = monotonic()
    entry = None if refresh else _CACHE.get(key_id)
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
        _CACHE.pop(key_id, None)
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

    A secret that does not match a cached entry is checked once more against
    a fresh read of the record when the entry is at least
    :data:`_REFRESH_SECONDS` old, so a freshly rotated key is accepted on
    every instance within one read instead of a whole cache window; a flood
    of wrong secrets against one key costs at most one read a second.

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
        unauthorized("Malformed tenant API key")
    key_id, secret = parsed
    entry = await _lookup(key_id, secret)
    if not _matches(secret, entry):
        if monotonic() - entry.fetched < _REFRESH_SECONDS:
            unauthorized("Invalid tenant API key")
        entry = await _lookup(key_id, secret, refresh=True)
        if not _matches(secret, entry):
            unauthorized("Invalid tenant API key")
    if entry.disabled:
        unauthorized(f"Tenant API key '{key_id}' is disabled")
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
        unauthorized("Malformed tenant key ID")
    entry = await _lookup(key_id, "")
    if entry.disabled:
        unauthorized(f"Tenant API key '{key_id}' is disabled")
    return entry.tenant


def _delivery_parameter(key_id: str) -> str:
    """Name the Parameter Store parameter one key is delivered through.

    Args:
        key_id: The key being delivered.

    Returns:
        The parameter name.
    """
    return f"{SETTINGS.tenant_key_ssm_parameter_prefix}/{key_id}"


def _secret_name(key_id: str) -> str:
    """Name the Secrets Manager secret one key is stored in.

    Args:
        key_id: The key being stored.

    Returns:
        The secret name.
    """
    return f"{SETTINGS.tenant_key_secretsmanager_prefix}/{key_id}"


def _store_target(key_id: str) -> str:
    """Name where one key is published, for the operator's log lines.

    Args:
        key_id: The key.

    Returns:
        The secret name, or the parameter name when keys are delivered
        through Parameter Store.
    """
    if SETTINGS.tenant_key_secretsmanager_prefix:
        return f"secret '{_secret_name(key_id)}'"
    return f"SSM parameter '{_delivery_parameter(key_id)}'"


def _version_token(key_id: str, rotation: int) -> str:
    """Derive the request token of one key's *rotation*-th secret version.

    Deterministic, so every instance writing the same rotation of the same
    key sends the same token: Secrets Manager then either accepts a repeat
    of the winning value or refuses a different one, and the loser reads the
    winner's version back by this same token.

    Args:
        key_id: The key.
        rotation: 0 for the minted version, then one more per rotation.

    Returns:
        The token, which is also the version's ID.
    """
    return str(uuid5(_TOKEN_NAMESPACE, f"{key_id}/{rotation}"))


def _full_key(key_id: str, secret: str) -> str:
    """Assemble the credential a tenant presents.

    Args:
        key_id: The key ID.
        secret: The secret.

    Returns:
        The full API key.
    """
    return f"{KEY_PREFIX}{key_id}-{secret}"


def _new_secret() -> str:
    """Draw a fresh key secret.

    Returns:
        The secret.
    """
    return "".join(choice(_ALPHABET) for _ in range(_SECRET_LENGTH))


def _error_code(error: ClientError) -> str:
    """Read the AWS error code off a client error.

    Args:
        error: The error.

    Returns:
        The code, or an empty string when AWS sent none.
    """
    return error.response.get("Error", {}).get("Code", "")


def _generation(item: Item, attribute: str) -> int:
    """Read a generation counter off a record.

    Args:
        item: The record.
        attribute: The counter's attribute name.

    Returns:
        The counter, or 0 when absent or not a number.
    """
    value = item.get(attribute)
    return value if isinstance(value, int) else 0


def _adopt(key_id: str, secret_name: str, value: str) -> str | None:
    """Take the key a secret version holds as this tenant's, if it is one.

    Args:
        key_id: The tenant the version must belong to.
        secret_name: The secret, for the operator's log line.
        value: The version's value.

    Returns:
        The key's secret, or None -- reported, never the value itself --
        when the version holds something that is not this tenant's key.
    """
    recovered = _parse(value)
    if recovered is None or recovered[0] != key_id:
        log_error_details(
            f"Secret '{secret_name}' does not hold tenant key '{key_id}': "
            "remove the offending version to let the server write one",
            level="warning",
        )
        return None
    return recovered[1]


async def _read_version(client: Any, secret_name: str, token: str) -> str:  # noqa: ANN401 - the client is untyped
    """Read the value of the version another write defined under *token*.

    Secrets Manager reads lag its writes, so a version just written by
    another instance may not be readable for a moment: the read is retried
    before the miss is taken for real.

    Args:
        client: The Secrets Manager client.
        secret_name: The secret.
        token: The version's ID, which is the request token that wrote it.

    Returns:
        The version's value.

    Raises:
        ClientError: The version could not be read within the retries.
    """
    for remaining in range(_READ_LAG_ATTEMPTS - 1, -1, -1):
        try:
            response = await client.get_secret_value(
                SecretId=secret_name, VersionId=token
            )
        except ClientError as error:
            if _error_code(error) != "ResourceNotFoundException" or not remaining:
                raise
            await sleep(_READ_LAG_SECONDS)
            continue
        value: str = response["SecretString"]
        return value
    raise AssertionError  # pragma: no cover - the loop always returns or raises


async def _put_version(
    client: Any,  # noqa: ANN401 - the client is untyped
    secret_name: str,
    token: str,
    value: str,
    *,
    pending: bool,
) -> str:
    """Write one secret version under a deterministic token, or read the one that won.

    Args:
        client: The Secrets Manager client.
        secret_name: The secret.
        token: The version's request token.
        value: The key to write.
        pending: Stage the version as ``AWSPENDING`` rather than current.

    Returns:
        The value the version holds: *value*, or the one another instance
        wrote first under the same token.

    Raises:
        ClientError: The version could not be written or read back.
    """
    stages = {"VersionStages": [_PENDING_STAGE]} if pending else {}
    try:
        await client.put_secret_value(
            SecretId=secret_name, ClientRequestToken=token, SecretString=value, **stages
        )
    except ClientError as error:
        if _error_code(error) != "ResourceExistsException":
            raise
        return await _read_version(client, secret_name, token)
    return value


def _store_kms_kwargs() -> dict[str, str]:
    """Build the encryption-key argument of a secret creation.

    Returns:
        ``KmsKeyId`` when a store key is configured, else nothing at all --
        which is what selects the AWS-managed key.
    """
    if SETTINGS.tenant_key_secretsmanager_kms_key_id:
        return {"KmsKeyId": SETTINGS.tenant_key_secretsmanager_kms_key_id}
    return {}


async def _store_key(key_id: str, name: str) -> str | None:
    """Store a freshly minted key as the current version of its tenant's secret.

    The create is create-once: the instance that created the secret defined
    the secret, and any other instance -- or a retry after a crash between
    storing and recording -- reads the current version back and records the
    same secret's hash. A container that already exists without a version,
    as the Terraform module pre-creates one, receives the key as its first
    version, which the service makes current.

    Args:
        key_id: The key to store.
        name: The tenant's declared name, for the secret description.

    Returns:
        The secret the current version holds, or None when the secret holds
        something that is not this tenant's key.

    Raises:
        ClientError: The secret could not be written or read back.
    """
    client = get_client("secretsmanager", _STORE_REGION)
    secret_name = _secret_name(key_id)
    token = _version_token(key_id, 0)
    value = _full_key(key_id, _new_secret())
    try:
        await client.create_secret(
            Name=secret_name,
            ClientRequestToken=token,
            SecretString=value,
            Description=f"stdapi.ai API key of tenant '{name}'",
            **_store_kms_kwargs(),
        )
    except ClientError as error:
        if _error_code(error) != "ResourceExistsException":
            raise
        try:
            response = await client.get_secret_value(
                SecretId=secret_name, VersionStage=_CURRENT_STAGE
            )
        except ClientError as read_error:
            if _error_code(read_error) != "ResourceNotFoundException":
                raise
            value = await _put_version(client, secret_name, token, value, pending=False)
        else:
            value = response["SecretString"]
    return _adopt(key_id, secret_name, value)


async def _deliver_key(key_id: str, name: str) -> str | None:
    """Deliver a freshly minted key once through Parameter Store.

    The Parameter Store write is create-once, which makes the whole mint
    idempotent: the instance that created the parameter defined the secret,
    and any other instance -- or a retry after a crash between delivery and
    recording -- reads the parameter back and records the same secret's hash.

    A genuine race is not answered with that create-once refusal, though:
    Parameter Store throttles concurrent writes of one name with
    :data:`_THROTTLED` before either write lands, so a loser has to retry to
    find the winner's parameter rather than take it for absent.

    Args:
        key_id: The key to deliver.
        name: The tenant's declared name, for the parameter description.

    Returns:
        The secret the parameter holds, or None when the parameter holds
        something that is not this tenant's key.

    Raises:
        ClientError: The parameter could not be written or read back.
    """
    parameter = _delivery_parameter(key_id)
    secret = _new_secret()
    ssm_client = get_client("ssm", _STORE_REGION)
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
                Value=_full_key(key_id, secret),
                Type="SecureString",
                Overwrite=False,
                Description=(
                    f"stdapi.ai API key of tenant '{name}'. "
                    "Deliver it to the tenant, then delete this parameter."
                ),
                **key_kwargs,
            )
        except ClientError as error:
            code = _error_code(error)
            if code == _THROTTLED and remaining:
                await sleep(_MINT_RETRY_SECONDS)
                continue
            if code != "ParameterAlreadyExists":
                raise
            value = (
                await ssm_client.get_parameter(Name=parameter, WithDecryption=True)
            )["Parameter"]["Value"]
            recovered = _parse(value)
            if recovered is None or recovered[0] != key_id:
                log_error_details(
                    f"SSM parameter '{parameter}' does not hold tenant key "
                    f"'{key_id}': delete the parameter to let the server mint one",
                    level="warning",
                )
                return None
            return recovered[1]
        return secret
    raise AssertionError  # pragma: no cover - the loop always returns or raises


async def _mint(key_id: str, name: str, tenant_item: Item) -> None:
    """Mint, publish and record the secret of one pending tenant key.

    Args:
        key_id: The key to mint.
        name: The tenant's declared name, for the store's description.
        tenant_item: The tenant record, whose ``key_generation`` the
            credential record starts at.

    Raises:
        ClientError: The key could not be published or read back.
        TableUnavailableError: The credential record could not be written.
    """
    if SETTINGS.tenant_key_secretsmanager_prefix:
        secret = await _store_key(key_id, name)
    else:
        secret = await _deliver_key(key_id, name)
    if secret is None:
        return
    salt = token_bytes(_SALT_SIZE)
    # Minted with every key so registering a role later needs no write.
    external_id = webuuid()
    record: Item = {
        PARTITION_KEY: _PARTITION,
        SORT_KEY: item_key(_SECRET_KIND, key_id),
        "secret_hash": _hash_secret(secret, salt),
        "salt": salt,
        "external_id": external_id,
        "minted_at": int(time()),
    }
    if generation := _generation(tenant_item, "key_generation"):
        record["generation"] = generation
    written = await put_item(record, condition=f"attribute_not_exists({PARTITION_KEY})")
    log_error_details(
        f"Minted tenant API key '{key_id}' into {_store_target(key_id)}, "
        "with an ExternalId for a cross-account role in its credential record"
        if written
        else f"Tenant API key '{key_id}' was minted by another instance",
        level="info",
    )


def _rotation_due(tenant_item: Item, secret_item: Item, now: float) -> bool:
    """Whether one key's rotation is due, or was started and not finished.

    Args:
        tenant_item: The operator-declared tenant record.
        secret_item: The server-minted credential record.
        now: The current epoch second.

    Returns:
        True when the record still carries a pending version, when the
        tenant's ``key_generation`` exceeds the recorded one, or when the
        key is older than ``tenant_key_rotation_days``.
    """
    if secret_item.get("pending_version_id") is not None:
        return True
    if _generation(tenant_item, "key_generation") > _generation(
        secret_item, "generation"
    ):
        return True
    days = SETTINGS.tenant_key_rotation_days
    if days is None:
        return False
    last = _generation(secret_item, "rotated_at") or _generation(
        secret_item, "minted_at"
    )
    return last + days * _DAY_SECONDS <= now


async def _put_pending(
    client: Any,  # noqa: ANN401 - the client is untyped
    key_id: str,
    secret_name: str,
    token: str,
) -> str:
    """Write a rotation's new key as the pending version of the tenant's secret.

    A key minted before the store was enabled has no secret yet: its
    container is created empty first, and the pending version then becomes
    the secret's first, and therefore current, version.

    Args:
        client: The Secrets Manager client.
        key_id: The key being rotated.
        secret_name: The tenant's secret.
        token: The version's request token.

    Returns:
        The value the pending version holds, this write's or the one another
        instance wrote first.

    Raises:
        ClientError: The version could not be written or read back.
    """
    value = _full_key(key_id, _new_secret())
    try:
        return await _put_version(client, secret_name, token, value, pending=True)
    except ClientError as error:
        if _error_code(error) != "ResourceNotFoundException":
            raise
    try:
        await client.create_secret(
            Name=secret_name,
            Description=f"stdapi.ai API key of tenant '{key_id}'",
            **_store_kms_kwargs(),
        )
    except ClientError as error:
        if _error_code(error) != "ResourceExistsException":
            raise
    return await _put_version(client, secret_name, token, value, pending=True)


async def _promote(client: Any, secret_name: str, token: str) -> bool:  # noqa: ANN401 - the client is untyped
    """Make the version written under *token* the secret's current one.

    Moving ``AWSCURRENT`` requires naming the version it leaves, which is
    read from the secret's label map right before each attempt: the move is
    then a no-op once the label is already there, and a stale name -- the
    label moved between the read and the move -- is refused by the service
    and simply re-read. The service moves ``AWSPREVIOUS`` by itself.

    Args:
        client: The Secrets Manager client.
        secret_name: The tenant's secret.
        token: The version's ID.

    Returns:
        True once the version is current and no longer pending, False when
        the service did not confirm the move within the retries.

    Raises:
        ClientError: A call failed for a reason other than a stale label.
    """
    for remaining in range(_READ_LAG_ATTEMPTS - 1, -1, -1):
        stages = (await client.describe_secret(SecretId=secret_name)).get(
            "VersionIdsToStages", {}
        )
        current = next(
            (version for version, labels in stages.items() if _CURRENT_STAGE in labels),
            None,
        )
        if current == token:
            break
        try:
            await client.update_secret_version_stage(
                SecretId=secret_name,
                VersionStage=_CURRENT_STAGE,
                MoveToVersionId=token,
                **({"RemoveFromVersionId": current} if current else {}),
            )
        except ClientError as error:
            if _error_code(error) != "InvalidParameterException":
                raise
        if remaining:
            await sleep(_READ_LAG_SECONDS)
    else:
        return False
    try:
        await client.update_secret_version_stage(
            SecretId=secret_name, VersionStage=_PENDING_STAGE, RemoveFromVersionId=token
        )
    except ClientError as error:
        # Already cleared by an earlier attempt, or never staged as pending.
        if _error_code(error) != "InvalidParameterException":
            raise
    return True


async def _rotate(key_id: str, tenant_item: Item, secret_item: Item) -> None:
    """Rotate one key, or finish a rotation a crash left half-done.

    Four steps, each idempotent: the new key is written as the pending
    version, recorded in the table with the superseded hash kept, promoted to
    current, and the record's pending marker cleared -- the one write that
    clears it being the one that starts the overlap. The table accepts the
    new key before it becomes current and keeps accepting the superseded one
    until the promotion has landed, so a client that re-reads the secret is
    never handed a key the gateway refuses, however long a promotion the
    store refuses stays open.

    Args:
        key_id: The key to rotate.
        tenant_item: The operator-declared tenant record.
        secret_item: The server-minted credential record, as read by the
            reconciliation.

    Raises:
        ClientError: A store call failed.
        TableUnavailableError: The credential record could not be written.
    """
    client = get_client("secretsmanager", _STORE_REGION)
    secret_name = _secret_name(key_id)
    token = secret_item.get("pending_version_id")
    rotated = secret_item
    if not isinstance(token, str):
        rotation = _generation(secret_item, "rotations") + 1
        token = _version_token(key_id, rotation)
        secret = _adopt(
            key_id, secret_name, await _put_pending(client, key_id, secret_name, token)
        )
        if secret is None:
            return
        now = int(time())
        salt = token_bytes(_SALT_SIZE)
        rotated = {
            name: value
            for name, value in secret_item.items()
            if not name.startswith("previous_")
        }
        rotated.update(
            secret_hash=_hash_secret(secret, salt),
            salt=salt,
            rotated_at=now,
            rotations=rotation,
            generation=max(
                _generation(secret_item, "generation"),
                _generation(tenant_item, "key_generation"),
            ),
            pending_version_id=token,
        )
        # Kept whatever the overlap is: until the promotion lands, the
        # superseded secret is the one the store still serves as current.
        rotated.update(
            previous_secret_hash=secret_item["secret_hash"],
            previous_salt=secret_item["salt"],
            previous_until=now + SETTINGS.tenant_key_rotation_overlap_seconds,
        )
        written = await put_item(
            rotated,
            condition="attribute_not_exists(rotations) OR rotations < :rotation",
            condition_values={":rotation": rotation},
        )
        if not written:
            log_error_details(
                f"Rotation {rotation} of tenant API key '{key_id}' was recorded "
                "by another instance",
                level="info",
            )
            return
        log_error_details(
            f"Rotated tenant API key '{key_id}': version '{token}' of secret "
            f"'{secret_name}' is recorded and accepted, promoting it to "
            f"{_CURRENT_STAGE}",
            level="info",
        )
    if not await _promote(client, secret_name, token):
        log_error_details(
            f"Version '{token}' of secret '{secret_name}' could not be confirmed "
            f"as {_CURRENT_STAGE} for tenant API key '{key_id}': the promotion is "
            "retried at the next reconciliation",
            level="warning",
        )
        return
    # One conditional write ends the rotation and starts the overlap, so the
    # superseded secret is refused from the moment it stops being served and
    # not one request earlier.
    promoted = {
        name: value
        for name, value in rotated.items()
        if name != "pending_version_id" and not name.startswith("previous_")
    }
    overlap = SETTINGS.tenant_key_rotation_overlap_seconds
    if overlap and "previous_secret_hash" in rotated:
        promoted.update(
            previous_secret_hash=rotated["previous_secret_hash"],
            previous_salt=rotated["previous_salt"],
            previous_until=int(time()) + overlap,
        )
    await put_item(
        promoted,
        condition="pending_version_id = :token",
        condition_values={":token": token},
    )
    log_error_details(
        f"Secret '{secret_name}' now serves version '{token}' as {_CURRENT_STAGE} "
        f"for tenant API key '{key_id}'",
        level="info",
    )


async def _rotate_due(key_id: str, tenant_item: Item, secret_item: Item) -> None:
    """Rotate one key, reporting rather than raising failures.

    Args:
        key_id: The key to rotate.
        tenant_item: Its tenant record.
        secret_item: Its credential record.
    """
    try:
        await _rotate(key_id, tenant_item, secret_item)
    except (ClientError, BotoCoreError, TableUnavailableError) as error:
        detail = (
            error.detail
            if isinstance(error, TableUnavailableError)
            else _store_failure_detail(key_id, error)
        )
        log_error_details(
            f"Tenant API key '{key_id}' could not be rotated: {detail}", level="warning"
        )
    except Exception as error:  # noqa: BLE001 -- one unrotatable key must not stop the others
        # A record missing an attribute this build reads would otherwise end the
        # whole pass, leaving every tenant listed after it unminted too.
        log_error_details(
            f"Tenant API key '{key_id}' could not be rotated: {type(error).__name__}",
            level="error",
        )


async def _scrub_previous(key_id: str, secret_item: Item) -> None:
    """Drop the superseded hash of a key whose overlap has ended.

    Conditional on the overlap deadline it read, so a rotation that started
    meanwhile is never overwritten. Failures are reported rather than raised;
    the next reconciliation retries.

    Args:
        key_id: The key.
        secret_item: Its credential record, as read by the reconciliation.
    """
    try:
        await put_item(
            {
                name: value
                for name, value in secret_item.items()
                if not name.startswith("previous_")
            },
            condition="previous_until = :until",
            condition_values={":until": secret_item["previous_until"]},
        )
    except (ClientError, BotoCoreError, TableUnavailableError) as error:
        detail = (
            error.detail
            if isinstance(error, TableUnavailableError)
            else type(error).__name__
        )
        log_error_details(
            f"The superseded secret of tenant key '{key_id}' could not be "
            f"dropped from its credential record: {detail}",
            level="warning",
        )


def _rotation_work(
    tenants: dict[str, Item], secrets: dict[str, Item]
) -> tuple[dict[str, tuple[Item, Item]], dict[str, Item]]:
    """Sort the minted keys into those due for rotation and those to scrub.

    Without the Secrets Manager store nothing rotates: a raised
    ``key_generation`` is then reported once per key and otherwise ignored.

    Args:
        tenants: The tenant records by key ID.
        secrets: The credential records by key ID.

    Returns:
        The keys due for rotation, with both records, and the keys whose
        superseded secret has expired, with their credential record.
    """
    due: dict[str, tuple[Item, Item]] = {}
    stale: dict[str, Item] = {}
    now = time()
    stored = bool(SETTINGS.tenant_key_secretsmanager_prefix)
    for key_id, secret_item in secrets.items():
        tenant_item = tenants.get(key_id)
        if (
            tenant_item is None
            or not _KEY_ID_RE(key_id)
            or not readable_schema(tenant_item)
            or not readable_schema(secret_item)
        ):
            continue
        if not stored:
            if (
                _generation(tenant_item, "key_generation")
                > _generation(secret_item, "generation")
                and f"rotate:{key_id}" not in _REPORTED
            ):
                _REPORTED.add(f"rotate:{key_id}")
                log_error_details(
                    f"The key_generation of tenant '{key_id}' is ignored: "
                    "rotating a key needs tenant_key_secretsmanager_prefix, "
                    "keys delivered through Parameter Store are never rotated",
                    level="warning",
                )
            continue
        if _rotation_due(tenant_item, secret_item, now):
            due[key_id] = (tenant_item, secret_item)
        elif (until := secret_item.get("previous_until")) is not None and (
            not isinstance(until, int) or until <= now
        ):
            stale[key_id] = secret_item
    return due, stale


async def _list_records() -> tuple[dict[str, Item], dict[str, Item], dict[str, str]]:
    """List every tenant-key record, sorted by kind.

    Returns:
        The tenant records and the credential records, each by key ID, and
        the sort key each credential record was read under -- so a record
        whose key ID is out of spec is still addressable without rebuilding
        its key.

    Raises:
        TableUnavailableError: The partition could not be listed.
    """
    tenants: dict[str, Item] = {}
    secrets: dict[str, Item] = {}
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
    return tenants, secrets, secret_sort_keys


async def reconcile_tenant_keys() -> None:
    """Mint, rotate and clean up every tenant key that needs it.

    A tenant record with no credential record is pending: the operator's
    tooling declared it and the secret does not exist yet. A credential record
    with no tenant record is orphaned: the tenant was destroyed and the hash
    is inert, so it is removed. A key stored in Secrets Manager is rotated
    when its schedule or its tenant's ``key_generation`` says so, and its
    superseded secret dropped once the overlap has ended.

    Raises:
        TableUnavailableError: The partition could not be listed.
    """
    tenants, secrets, secret_sort_keys = await _list_records()
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
    due, stale = _rotation_work(tenants, secrets)
    # Rotating or scrubbing re-puts the whole record from the copy listed above,
    # which would erase an ExternalId the backfill writes in this same pass.
    due = {
        key_id: work for key_id, work in due.items() if key_id not in unminted_external
    }
    stale = {
        key_id: item
        for key_id, item in stale.items()
        if key_id not in unminted_external
    }
    if not (pending or orphans or unminted_external or due or stale):
        return
    with log_background_event("tenant_keys_reconcile", webuuid()):
        for key_id, item in pending.items():
            await _mint_pending(key_id, item)
        for key_id, item in unminted_external.items():
            await _backfill_external_id(key_id, item)
        for key_id, (tenant_item, secret_item) in due.items():
            await _rotate_due(key_id, tenant_item, secret_item)
        for key_id, item in stale.items():
            await _scrub_previous(key_id, item)
        for key_id, sort_key in orphans.items():
            await _drop_orphan(key_id, sort_key)


async def _drop_orphan(key_id: str, sort_key: str) -> None:
    """Remove the credential record of a tenant that no longer exists.

    Deleting one revokes a credential, so the tenant record is re-read
    immediately before: a tooling-driven destroy-and-recreate would otherwise
    let a pass started inside that gap revoke a key already delivered.

    The secret a stored key lives in is left to whoever declared the tenant:
    the Terraform module destroys it with the tenant record, and a deletion
    racing that one would fail the module's own. Its value is inert once the
    record is gone, and the revocation names it so nothing is forgotten.

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
    leftover = (
        f"; delete its secret '{_secret_name(key_id)}' if it still exists"
        if SETTINGS.tenant_key_secretsmanager_prefix
        else ""
    )
    log_error_details(
        f"Revoked the credential record of destroyed tenant key '{key_id}'{leftover}",
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


def _store_failure_detail(key_id: str, error: ClientError | BotoCoreError) -> str:
    """Describe why publishing one tenant key failed, for the operator log.

    Reuses :func:`stdapi.api_errors.iam_denial_detail`, so a missing permission
    reads the same as every other AWS denial the server reports -- including
    one on the KMS key encrypting the secret or parameter, which fails the
    same store call and is named by its own action rather than guessed at.

    Args:
        key_id: The key that could not be published.
        error: The failure raised writing or reading back the key.

    Returns:
        Which call failed, on which secret or parameter, and why -- the
        missing IAM permission when AWS denied it, else the bare error code.
        Never the AWS error's own message text, which may name the caller's
        principal.
    """
    target = _store_target(key_id)
    if not isinstance(error, ClientError):
        return f"writing {target} could not be sent ({type(error).__name__})"
    where = f"{error.operation_name} on {target}"
    if denial := iam_denial_detail(error):
        return f"{where} was denied: {denial}"
    code = _error_code(error)
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
        await _mint(key_id, name if isinstance(name, str) else key_id, item)
    except (ClientError, BotoCoreError, TableUnavailableError) as error:
        detail = (
            error.detail
            if isinstance(error, TableUnavailableError)
            else _store_failure_detail(key_id, error)
        )
        log_error_details(
            f"Tenant API key '{key_id}' could not be minted: {detail}", level="warning"
        )


def _foreign_region_key(key_id: str | None) -> bool:
    """Whether a KMS key reference names a key of another region.

    Args:
        key_id: The key reference, as configured.

    Returns:
        True when it is an ARN whose region is not the store's.
    """
    return bool(
        key_id and key_id.startswith("arn:") and key_id.split(":")[3] != _STORE_REGION
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
    # Both stores are regional and cannot encrypt with a key from elsewhere.
    # Warned rather than refused, and checked here rather than beside the shape
    # validations in config.py: the region is only detected once SETTINGS exists,
    # and refusing would take a whole deployment down over one optional store
    # setting. A mint failure names the KMS permission the call was denied.
    if SETTINGS.tenant_key_secretsmanager_prefix:
        key_id, what = SETTINGS.tenant_key_secretsmanager_kms_key_id, "stored"
    else:
        key_id, what = SETTINGS.tenant_key_ssm_kms_key_id, "delivered"
    if _foreign_region_key(key_id):
        add_server_warning(
            start_event,
            f"Tenant API keys cannot be {what}: their KMS key '{key_id}' is "
            f"not in region '{_STORE_REGION}'",
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
