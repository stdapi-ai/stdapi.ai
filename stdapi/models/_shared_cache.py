"""The model list a deployment's servers publish to each other.

Optional, and off unless ``model_cache_shared`` and ``aws_dynamodb_table`` are
both set. Without it every server discovers the catalogue on its own, which is
what this replaces: one server sweeps AWS Bedrock, publishes what it found, and
the rest read it.

Three kinds of record live under ``pk=MODELCACHE#<fingerprint>``:

- ``sk=manifest`` -- the version, age, shard count and checksum of the
  published list. Written **last**, which is what makes a torn write
  unreadable rather than wrong: a reader only ever fetches the shards of the
  version its manifest names, and a half-written set has no manifest pointing
  at it yet.
- ``sk=shard#<version>#<n>`` -- the list itself, JSON, zstd-compressed and cut
  into pieces small enough for one item. Shards outlive their manifest by
  :data:`_SHARD_GRACE_SECONDS` so a manifest never points at a piece that has
  expired, and orphans from an abandoned version are deleted by the table's
  time-to-live rather than by a call anyone pays for.
- ``sk=lease`` -- held by the server currently sweeping, so a fleet performs one
  sweep rather than one each. It carries its own expiry, so a holder that dies
  mid-sweep costs one lease period rather than wedging the fleet.

The **fingerprint** is what keeps two servers from consuming each other's list:
it covers the server version, the AWS account and every ``aws_bedrock*``
setting -- deliberately every one of them, rather than the subset discovery
reads today, because being too broad only costs a refresh while being too
narrow serves one deployment's catalogue to another. A server that does not
recognise the fingerprint simply finds no cache. The record *layout* is carried
by the table's own schema attribute instead, so a newer build's records are
skipped rather than misread.

Nothing here is allowed to fail a request. Every table error becomes an
operator warning and a ``None``/``False`` answer, and the caller falls back to
sweeping AWS Bedrock exactly as a deployment without the table does.
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from secrets import token_hex
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic_core import from_json

from stdapi import server
from stdapi.aws import AWS_ENVIRONMENT
from stdapi.aws_dynamodb import (
    EXPIRES_AT_ATTRIBUTE,
    KEY_SEPARATOR,
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
from stdapi.config import SETTINGS
from stdapi.monitoring import add_server_warning, log_error_details
from stdapi.server import SERVER_VERSION
from stdapi.utils import to_json_bytes

if TYPE_CHECKING:
    from pydantic import JsonValue

    from stdapi.aws_dynamodb import Item
    from stdapi.monitoring import EventLog


#: Partition-key namespace of every shared model-list record.
NAMESPACE: Final = "MODELCACHE"

#: Record naming the shards, the version and the age of the published list.
_MANIFEST: Final = "manifest"

#: Record held by the server currently sweeping AWS Bedrock.
_LEASE: Final = "lease"

#: Record kind holding one compressed piece of the published list.
_SHARD: Final = "shard"

#: Attribute naming the server holding the lease.
_HOLDER_ATTRIBUTE: Final = "lease_holder"

#: Compressed bytes per shard, under DynamoDB's 400 KB item limit.
_SHARD_BYTES: Final = 350_000

#: Shards one list may span before publishing it is abandoned as a defect.
_MAX_SHARDS: Final = 32

#: Bytes a published list may decompress to, bounding an unauthenticated blob.
_MAX_PAYLOAD_BYTES: Final = 64_000_000

#: Seconds a sweeping server holds the lease before another may reclaim it.
LEASE_SECONDS: Final = 120

#: zstd level: the list is text, and this is where its ratio stops improving.
_COMPRESSION_LEVEL: Final = 10

#: Seconds shards outlive their manifest, so none expires while it is named.
_SHARD_GRACE_SECONDS: Final = 3600

#: Age at which a manifest is dropped even when nothing has refreshed it.
_MIN_MANIFEST_TTL_SECONDS: Final = 3600

#: Seconds a manifest's created_at may sit ahead of this server's clock.
_MAX_CLOCK_SKEW_SECONDS: Final = 5

#: Hexadecimal characters kept from the fingerprint digest.
_FINGERPRINT_LENGTH: Final = 32

#: Prefix of every setting that shapes what a discovery sweep returns.
_SHAPING_PREFIX: Final = "aws_bedrock"

#: Warnings already reported, so a failing table does not flood the log.
_REPORTED: Final[set[str]] = set()


class PublishedCatalog(NamedTuple):
    """A model list read back from the table.

    Attributes:
        payload: The decoded list, in the shape its publisher wrote.
        created_at: When the publishing server finished its sweep, which is the
            age the reader inherits rather than restarting from now.
    """

    payload: JsonValue
    created_at: int


def enabled() -> bool:
    """Whether this deployment publishes its model list.

    Returns:
        True when the feature is on and a table is configured for it.
    """
    return SETTINGS.model_cache_shared and bool(SETTINGS.aws_dynamodb_table)


def fingerprint() -> str:
    """Identify the servers whose model list is interchangeable with this one's.

    Returns:
        A digest of the record layout, the server version, the AWS account and
        every discovery-shaping setting.
    """
    shaping = {
        name: getattr(SETTINGS, name)
        for name in type(SETTINGS).model_fields
        if name.startswith(_SHAPING_PREFIX)
    }
    digest = sha256(
        to_json_bytes([SERVER_VERSION, AWS_ENVIRONMENT.get("account_id", ""), shaping])
    ).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


def _partition() -> str:
    """Return the partition every record of this server's list lives under.

    Returns:
        The partition key.
    """
    return item_key(NAMESPACE, fingerprint())


def _warn(start_event: EventLog | None, detail: str) -> None:
    """Report to the operator that the shared list is not working.

    Reported once per detail while the table keeps failing that way: an
    unreachable table is unreachable on every refresh, and repeating it buries
    the rest of the log. `_note_reachable` clears the record once the table
    answers again, so a failure that comes back is reported again rather than
    silently swallowed for the life of the process.

    Args:
        start_event: Startup event log to record the warning on, if any.
        detail: What failed, named for the operator.
    """
    if detail in _REPORTED:
        return
    _REPORTED.add(detail)
    if start_event is not None:
        add_server_warning(start_event, detail)
    else:
        log_error_details(detail, level="warning")


def _note_reachable() -> None:
    """Forget the failures reported so far, the table having answered again.

    Without this a fault that clears and later returns is never reported a
    second time, and the operator reads a healthy log while every refresh
    silently falls back to a sweep of this server's own.
    """
    _REPORTED.clear()


async def read_catalog(  # noqa: PLR0911 - one arm per way a manifest is unusable
    start_event: EventLog | None,
) -> PublishedCatalog | None:
    """Read the published model list, when one is current.

    Args:
        start_event: Startup event log to record warnings on, if any.

    Returns:
        The published list, or None when there is none this server can use --
        no manifest, a layout it cannot read, a list already older than
        ``model_cache_seconds``, a manifest naming more pieces or more bytes
        than one list is ever cut into, a shard set that does not match its
        manifest, or a table that did not answer. Every one of those is a cache
        miss, and the caller sweeps AWS Bedrock instead.
    """
    partition = _partition()
    try:
        manifest = await get_item(partition, _MANIFEST, consistent=True)
    except TableUnavailableError as unavailable:
        _warn(start_event, unavailable.detail)
        return None
    if manifest is None or not readable_schema(manifest):
        return None
    version = manifest.get("version")
    created_at = manifest.get("created_at")
    shard_count = manifest.get("shard_count")
    checksum = manifest.get("checksum")
    if (
        not isinstance(version, str)
        or not isinstance(created_at, int)
        or not isinstance(shard_count, int)
        or not isinstance(checksum, str)
    ):
        return None
    # Checked before the pieces are fetched: nothing this server publishes ever
    # spans more, so a manifest that claims to is a record not to act on.
    if not 0 < shard_count <= _MAX_SHARDS:
        _warn(start_event, _corrupt(f"names {shard_count} pieces"))
        return None
    # The manifest's own age is the single staleness rule: the table's
    # time-to-live deletes an expired item eventually, never on time.
    age = SETTINGS.now().timestamp() - created_at
    if age >= SETTINGS.model_cache_seconds:
        return None
    # A created_at ahead of this clock past a small tolerance is a corrupt or
    # hostile record, not a peer to trust: unbounded, it would keep this
    # server serving whatever it names until the ceiling never fires.
    if age < -_MAX_CLOCK_SKEW_SECONDS:
        _warn(start_event, _corrupt("has a created_at in the future"))
        return None
    payload = await _read_shards(partition, version, shard_count, checksum, start_event)
    return None if payload is None else PublishedCatalog(payload, created_at)


async def _read_shards(  # noqa: PLR0911 - one arm per way a shard set is unusable
    partition: str,
    version: str,
    shard_count: int,
    checksum: str,
    start_event: EventLog | None,
) -> JsonValue | None:
    """Read back the pieces of one published version and decode them.

    Args:
        partition: The partition the pieces live under.
        version: The version its manifest names.
        shard_count: How many pieces that version was cut into.
        checksum: Digest of the compressed list, before it was cut up.
        start_event: Startup event log to record warnings on, if any.

    Returns:
        The decoded list, or None when the pieces are incomplete, do not add up
        to what the manifest describes, or cannot be decoded.
    """
    # Imported here: compression is needed only by a deployment that shares.
    from compression.zstd import ZstdDecompressor, ZstdError  # noqa: PLC0415

    try:
        shards = await query_partition(
            partition,
            sort_key_prefix=item_key(_SHARD, version) + KEY_SEPARATOR,
            consistent=True,
        )
    except TableUnavailableError as unavailable:
        _warn(start_event, unavailable.detail)
        return None
    if len(shards) != shard_count:
        # A torn write, or a piece the time-to-live reached first.
        return None
    blob = bytearray()
    for shard in shards:
        data = shard.get("data")
        if not readable_schema(shard) or not isinstance(data, bytes):
            return None
        blob += data
    if sha256(blob).hexdigest() != checksum:
        _warn(start_event, _corrupt("does not match its own checksum"))
        return None
    # Bounded, because the blob is expanded before anything has established
    # where it came from: a few hundred kilobytes of zstd decompress to
    # gigabytes of repetitive JSON, which would take the fleet down at once.
    decompressor = ZstdDecompressor()
    try:
        decoded = decompressor.decompress(bytes(blob), max_length=_MAX_PAYLOAD_BYTES)
    except ZstdError as exception:
        _warn(start_event, _corrupt(f"could not be decoded ({exception})"))
        return None
    if not decompressor.eof:
        _warn(
            start_event,
            _corrupt(
                "stops short of its own end or expands past the "
                f"{_MAX_PAYLOAD_BYTES} bytes a published list may occupy"
            ),
        )
        return None
    try:
        payload: JsonValue = from_json(decoded)
    except ValueError as exception:
        _warn(start_event, _corrupt(f"could not be decoded ({exception})"))
        return None
    return payload


def _corrupt(what: str) -> str:
    """Describe an unusable published list for the operator.

    Args:
        what: What is wrong with it.

    Returns:
        The warning detail.
    """
    return (
        f"The shared model list in DynamoDB table "
        f"'{SETTINGS.aws_dynamodb_table}' {what} and was ignored; this server "
        "discovered the models itself."
    )


async def publish_catalog(
    payload: JsonValue, created_at: int, start_event: EventLog | None
) -> None:
    """Publish a freshly swept model list for the other servers to read.

    Shards first, manifest last: until the manifest names the new version, a
    reader keeps using the previous one, so no reader ever sees a half-written
    list. The previous version's shards are left to the table's time-to-live
    rather than deleted, which costs nothing and cannot race a reader that is
    still fetching them.

    Args:
        payload: The list to publish, JSON-encodable.
        created_at: When the sweep that produced it finished, epoch seconds.
        start_event: Startup event log to record warnings on, if any.
    """
    # Imported here: compression is needed only by a deployment that shares.
    from compression.zstd import compress  # noqa: PLC0415

    blob = compress(to_json_bytes(payload), level=_COMPRESSION_LEVEL)
    shards = [blob[at : at + _SHARD_BYTES] for at in range(0, len(blob), _SHARD_BYTES)]
    if len(shards) > _MAX_SHARDS:
        _warn(
            start_event,
            f"The model list compresses to {len(blob)} bytes, more than the "
            f"{_MAX_SHARDS * _SHARD_BYTES} bytes 'model_cache_shared' "
            "publishes; this server's list was not shared.",
        )
        return
    partition = _partition()
    version = token_hex(8)
    # ``model_cache_seconds`` is part of the floor because it is the age a
    # reader accepts a manifest until: expiring one sooner would silently stop
    # the sharing this feature exists for.
    manifest_expiry = created_at + max(
        SETTINGS.model_cache_max_stale_seconds,
        SETTINGS.model_cache_seconds,
        _MIN_MANIFEST_TTL_SECONDS,
    )
    try:
        for index, shard in enumerate(shards):
            await put_item(
                {
                    PARTITION_KEY: partition,
                    SORT_KEY: item_key(_SHARD, version, f"{index:04d}"),
                    "data": shard,
                    EXPIRES_AT_ATTRIBUTE: manifest_expiry + _SHARD_GRACE_SECONDS,
                }
            )
        await put_item(
            {
                PARTITION_KEY: partition,
                SORT_KEY: _MANIFEST,
                "version": version,
                "created_at": created_at,
                "shard_count": len(shards),
                "checksum": sha256(blob).hexdigest(),
                EXPIRES_AT_ATTRIBUTE: manifest_expiry,
            }
        )
    except TableUnavailableError as unavailable:
        _warn(start_event, unavailable.detail)


class Lease(StrEnum):
    """Outcome of claiming the sweep, which decides what the caller does next."""

    #: This server sweeps and publishes for the whole deployment.
    HELD = "held"
    #: A peer is sweeping; wait for what it publishes rather than duplicating it.
    PEER = "peer"
    #: The table did not answer, so no peer can be relied on to sweep either.
    UNAVAILABLE = "unavailable"


async def acquire_lease(start_event: EventLog | None) -> Lease:
    """Claim the right to be the server that sweeps AWS Bedrock this round.

    A table that did not answer is told apart from a peer holding the lease:
    both stop this server publishing, but only the peer is going to produce a
    catalog, so an unavailable table must not stop it serving itself.

    Args:
        start_event: Startup event log to record warnings on, if any.

    Returns:
        Which of the three outcomes occurred.
    """
    now = int(SETTINGS.now().timestamp())
    try:
        held = await put_item(
            {
                PARTITION_KEY: _partition(),
                SORT_KEY: _LEASE,
                _HOLDER_ATTRIBUTE: server.SERVER_NAME,
                EXPIRES_AT_ATTRIBUTE: now + LEASE_SECONDS,
            },
            condition=(f"attribute_not_exists({PARTITION_KEY}) OR #expires_at < :now"),
            condition_values={":now": now},
            condition_names={"#expires_at": EXPIRES_AT_ATTRIBUTE},
        )
    except TableUnavailableError as unavailable:
        _warn(start_event, unavailable.detail)
        return Lease.UNAVAILABLE
    # Reached on every refresh round, whoever won, so it is where a table that
    # has come back is noticed.
    _note_reachable()
    return Lease.HELD if held else Lease.PEER


async def release_lease(start_event: EventLog | None) -> None:
    """Give up the lease this server holds, at the end of its sweep.

    Conditional on still being the holder: a sweep that overran its lease may
    have been replaced by another server, and deleting that server's lease
    would let a third one start a second sweep.

    Args:
        start_event: Startup event log to record warnings on, if any.
    """
    condition_values: Item = {":holder": server.SERVER_NAME}
    try:
        await delete_item(
            _partition(),
            _LEASE,
            condition=f"{_HOLDER_ATTRIBUTE} = :holder",
            condition_values=condition_values,
        )
    except TableUnavailableError as unavailable:
        _warn(start_event, unavailable.detail)
