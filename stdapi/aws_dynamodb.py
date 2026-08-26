"""Amazon DynamoDB table access, shared by every feature that needs one.

One table serves every feature: the records are tiny, their key spaces are
disjoint, and a table per feature would multiply the setting, the Terraform
variable, the IAM statement and the sandbox resource for no isolation gain.

Its primary key is composite -- ``pk`` (S) plus ``sk`` (S) -- because a sort key
cannot be added to a table that already exists. Every key is built by
:func:`item_key`, which forbids the separator inside a part, so two namespaces
can never produce the same key:

- Tenant API keys: ``pk=TENANT``, ``sk=tenant#<key id>`` for the
  operator-declared record and ``sk=secret#<key id>`` for the server-minted
  credential hash -- one partition, so one query lists every tenant.
- Shared model cache: ``pk=MODELCACHE#<fingerprint>``, ``sk`` one of
  ``manifest``, ``lease`` or ``shard#<version>#<n>``.

A ``sk`` is the *kind* of record within its partition: a bare token for a kind
that has one record per partition, and ``<kind>#<discriminator>`` for a kind
that has many, so one ``begins_with`` query reads exactly one kind.

Every item carries :data:`SCHEMA_VERSION` under :data:`SCHEMA_ATTRIBUTE`, and
:data:`EXPIRES_AT_ATTRIBUTE` is reserved for the table's time-to-live: an item
that sets it is deleted when it expires, free of write throughput.

Nothing here decides what a failure means. Every call that does not complete
raises :class:`TableUnavailableError` carrying what the *operator* must fix, and
each feature applies its own policy to it -- refusing the request, or falling
back to what it does without the table.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Any, Final

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.monitoring import add_server_warning

if TYPE_CHECKING:
    from collections.abc import Mapping

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.monitoring import EventLog

    #: A value an item attribute can hold, in the subset DynamoDB and this module share.
    type ItemValue = (
        str
        | int
        | float
        | bool
        | bytes
        | list[ItemValue]
        | dict[str, ItemValue]
        | set[str]
        | None
    )

    #: A whole item, attribute name to value, as callers read and write it.
    type Item = dict[str, ItemValue]


#: Attribute holding the partition key, "<NAMESPACE>#<identifier>".
PARTITION_KEY: Final = "pk"

#: Attribute holding the sort key, the kind of record within its partition.
SORT_KEY: Final = "sk"

#: Separator between the parts of a key; forbidden inside a part.
KEY_SEPARATOR: Final = "#"

#: Attribute naming the item layout its writer used.
SCHEMA_ATTRIBUTE: Final = "schema"

#: Item layout this build writes, and the highest one it can read.
SCHEMA_VERSION: Final = 1

#: Attribute the table's time-to-live reads, as epoch seconds.
EXPIRES_AT_ATTRIBUTE: Final = "expires_at"

#: Region serving the table, which is a single regional resource with no failover.
TABLE_REGION: RegionName = (
    SETTINGS.aws_dynamodb_region or SETTINGS.aws_bedrock_regions[0]
)

#: Error code answering a conditional write whose condition did not hold.
_CONDITION_FAILED: Final = "ConditionalCheckFailedException"

#: Error code answering a call naming a table that does not exist in the region.
_TABLE_NOT_FOUND: Final = "ResourceNotFoundException"

#: Error codes answering a call the server's role is not allowed to make.
_ACCESS_DENIED_CODES: Final[frozenset[str]] = frozenset(
    {"AccessDenied", "AccessDeniedException", "UnrecognizedClientException"}
)

#: Time-to-live status of a table whose expiring items are actually deleted.
_TTL_ENABLED: Final = "ENABLED"


class TableUnavailableError(Exception):
    """A table call that did not complete, with what the operator must fix.

    Never rendered to a caller: its message names the table, the IAM action and
    the setting behind the failure, which is exactly what must not leak. Every
    feature catches it and answers in its own terms -- refusing the request as a
    feature this deployment cannot run, or continuing without the table.

    Attributes:
        detail: What failed, named for the operator, for a ``warning`` log line.
    """

    __slots__ = ("detail",)

    def __init__(self, detail: str) -> None:
        """Report a table call that did not complete.

        Args:
            detail: What failed, named for the operator: the IAM action, the
                table, the region and the setting.
        """
        self.detail = detail
        super().__init__(detail)


def item_key(*parts: str) -> str:
    """Build a partition or sort key from its parts.

    Args:
        *parts: The namespace or record kind, then whatever identifies the item
            within it. At least one, none empty, none containing
            :data:`KEY_SEPARATOR`.

    Returns:
        The parts joined by :data:`KEY_SEPARATOR`.

    Raises:
        ValueError: A part is empty or carries the separator, either of which
            would let two different items resolve to the same key.
    """
    if not parts:
        msg = "A DynamoDB key needs at least one part."
        raise ValueError(msg)
    for part in parts:
        if not part:
            msg = "A DynamoDB key part must not be empty."
            raise ValueError(msg)
        if KEY_SEPARATOR in part:
            msg = (
                f"A DynamoDB key part must not contain '{KEY_SEPARATOR}': "
                f"{part!r} would collide with another namespace."
            )
            raise ValueError(msg)
    return KEY_SEPARATOR.join(parts)


def readable_schema(item: Item) -> bool:
    """Whether this build can interpret *item*'s layout.

    An item written by a newer build during a rolling deployment may hold
    attributes this one does not understand. What to do about it is the
    feature's own decision: serving no cache is right for a cache, and refusing
    a request is right for a credential.

    Args:
        item: A decoded item.

    Returns:
        True when the item's layout is this build's or older.
    """
    schema = item.get(SCHEMA_ATTRIBUTE, SCHEMA_VERSION)
    return isinstance(schema, int) and schema <= SCHEMA_VERSION


def encode_value(value: ItemValue) -> dict[str, Any]:  # noqa: C901, PLR0911 - one arm per attribute type
    """Encode a Python value as a DynamoDB attribute value.

    Args:
        value: The value to encode.

    Returns:
        The single-entry ``{type: value}`` mapping DynamoDB expects.

    Raises:
        ValueError: The value is a number DynamoDB cannot hold, or an empty
            set, which DynamoDB rejects.
        TypeError: The value is of a type this table does not store.
    """
    match value:
        # Before int: bool is a subclass of it.
        case bool():
            return {"BOOL": value}
        case str():
            return {"S": value}
        case int():
            return {"N": str(value)}
        case float():
            if not isfinite(value):
                msg = "DynamoDB holds no NaN and no infinity."
                raise ValueError(msg)
            return {"N": repr(value)}
        case bytes() | bytearray() | memoryview():
            return {"B": bytes(value)}
        case None:
            return {"NULL": True}
        case list():
            return {"L": [encode_value(item) for item in value]}
        case dict():
            return {"M": {name: encode_value(item) for name, item in value.items()}}
        case set() | frozenset():
            if not value:
                msg = "DynamoDB holds no empty set; store an empty list instead."
                raise ValueError(msg)
            # Sorted so the same set always serializes identically, which keeps
            # a checksum over an encoded item stable.
            return {"SS": sorted(value)}
        case _:
            msg = f"{type(value).__name__} is not stored in this table."
            raise TypeError(msg)


def decode_value(value: Mapping[str, Any]) -> ItemValue:  # noqa: PLR0911 - one arm per attribute type
    """Decode a DynamoDB attribute value.

    A number comes back as an ``int`` unless the decimal string DynamoDB
    returns carries a fractional part or an exponent, which is not how it was
    written: the service stores every number as a decimal and trims its
    trailing zeroes, so ``1.0`` reads back as the ``int`` 1 and a caller that
    needs a ``float`` coerces one.

    Args:
        value: The single-entry ``{type: value}`` mapping DynamoDB returned.

    Returns:
        The decoded value.

    Raises:
        ValueError: The attribute carries a type this table does not store.
    """
    kind, raw = next(iter(value.items()))
    match kind:
        case "S" | "BOOL":
            return raw  # type: ignore[no-any-return]
        case "N":
            return float(raw) if any(c in raw for c in ".eE") else int(raw)
        case "B":
            return bytes(raw)
        case "NULL":
            return None
        case "L":
            return [decode_value(item) for item in raw]
        case "M":
            return {name: decode_value(item) for name, item in raw.items()}
        case "SS":
            return set(raw)
        case _:
            msg = f"Attribute type '{kind}' is not stored in this table."
            raise ValueError(msg)


def encode_item(item: Item) -> dict[str, dict[str, Any]]:
    """Encode a whole item.

    Args:
        item: Attribute names to values.

    Returns:
        The item in DynamoDB's attribute-value form.
    """
    return {name: encode_value(value) for name, value in item.items()}


def decode_item(item: Mapping[str, Mapping[str, Any]]) -> Item:
    """Decode a whole item.

    Args:
        item: The item in DynamoDB's attribute-value form.

    Returns:
        Attribute names to values.
    """
    return {name: decode_value(value) for name, value in item.items()}


def table_client_specs() -> tuple[tuple[str, RegionName | None], ...]:
    """Return the client specs the connection pool needs for the table.

    Returns:
        One ``("dynamodb", region)`` spec when a table is configured, and
        nothing at all otherwise -- a deployment that configures none opens no
        client, makes no call, and creates nothing to be billed for.
    """
    return (("dynamodb", SETTINGS.aws_dynamodb_region),) if _table() else ()


def _table() -> str | None:
    """Return the configured table name, if any.

    Returns:
        The table name, or None when no feature has been given one.
    """
    return SETTINGS.aws_dynamodb_table


def _client() -> Any:  # noqa: ANN401
    """Return the pooled DynamoDB client.

    Returns:
        The client serving :data:`TABLE_REGION`.

    Raises:
        TableUnavailableError: No table is configured, so the pool holds no
            client -- a feature called this without checking its own setting --
            or the pool holds no client for the table's region.
    """
    if not _table():
        msg = (
            "'aws_dynamodb_table' is not set: set it to the name of the "
            "DynamoDB table this deployment's shared records live in."
        )
        raise TableUnavailableError(msg)
    try:
        # Named literally: the container suite derives the botocore service data
        # its images must keep from the service names the source spells out.
        return get_client("dynamodb", SETTINGS.aws_dynamodb_region)
    except KeyError:
        msg = (
            f"No DynamoDB client is open for region {TABLE_REGION}: "
            "'aws_dynamodb_table' was set after the server started, or the "
            "server's AWS client pool did not finish starting, or it is "
            "shutting down."
        )
        raise TableUnavailableError(msg) from None


def _failure(
    error: BotoCoreError | ClientError, operation: str
) -> TableUnavailableError:
    """Describe a failed table call for the operator.

    Args:
        error: The AWS error, which never reaches a caller.
        operation: DynamoDB operation name, as IAM spells it.

    Returns:
        The error each feature applies its own policy to.
    """
    action = f"dynamodb:{operation}"
    table = _table()
    where = f"table '{table}' in {TABLE_REGION}"
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        if code in _ACCESS_DENIED_CODES:
            detail = (
                f"Access denied calling {action} on {where}: grant the server "
                f"role '{action}' on that table."
            )
        elif code == _TABLE_NOT_FOUND:
            detail = (
                f"{action} found no {where}: check that 'aws_dynamodb_table' "
                "names an existing table and that 'aws_dynamodb_region' names "
                "the region holding it."
            )
        else:
            detail = f"{action} on {where} failed ({code})."
        return TableUnavailableError(detail)
    return TableUnavailableError(
        f"{action} on {where} could not be sent ({type(error).__name__})."
    )


def _decode_failure(error: ValueError, operation: str) -> TableUnavailableError:
    """Describe an item this build cannot decode, for the operator.

    A newer build's item is skipped by :func:`readable_schema`, applied by the
    caller after decoding; an item carrying an attribute type this table does
    not store at all is a different problem decoding cannot recover from.

    Args:
        error: What :func:`decode_item` raised.
        operation: DynamoDB operation that returned the item, as IAM spells it.

    Returns:
        The error each feature applies its own policy to.
    """
    action = f"dynamodb:{operation}"
    where = f"table '{_table()}' in {TABLE_REGION}"
    return TableUnavailableError(
        f"{action} on {where} returned an item this build cannot decode: {error}"
    )


async def get_item(
    partition_key: str, sort_key: str, *, consistent: bool = False
) -> Item | None:
    """Read one item.

    Args:
        partition_key: The item's ``pk``.
        sort_key: The item's ``sk``.
        consistent: Read the latest write rather than an eventually consistent
            copy, at twice the read cost.

    Returns:
        The decoded item, or None when the table holds none under that key.

    Raises:
        TableUnavailableError: The read did not complete.
    """
    try:
        response = await _client().get_item(
            TableName=_table(),
            Key=encode_item({PARTITION_KEY: partition_key, SORT_KEY: sort_key}),
            ConsistentRead=consistent,
        )
    except (BotoCoreError, ClientError) as error:
        raise _failure(error, "GetItem") from error
    item = response.get("Item")
    if not item:
        return None
    try:
        return decode_item(item)
    except ValueError as error:
        raise _decode_failure(error, "GetItem") from error


async def put_item(
    item: Item,
    *,
    condition: str | None = None,
    condition_values: Item | None = None,
    condition_names: Mapping[str, str] | None = None,
) -> bool:
    """Write one item, whole, optionally only if a condition holds.

    The conditional form is what a lease and a create-once record are built
    from: the write and its test are one atomic operation, so two servers
    racing for the same key produce exactly one winner.

    Args:
        item: The item, carrying its own ``pk`` and ``sk``. Its schema version
            is stamped here unless the caller set one.
        condition: A DynamoDB condition expression the write requires, e.g.
            ``"attribute_not_exists(pk) OR expires_at < :now"``.
        condition_values: Values the condition's ``:`` placeholders stand for.
        condition_names: Names the condition's ``#`` placeholders stand for,
            for attributes whose name DynamoDB reserves.

    Returns:
        True when the item was written, False when the condition did not hold.

    Raises:
        ValueError: The item carries no ``pk`` or no ``sk``.
        TableUnavailableError: The write did not complete.
    """
    if PARTITION_KEY not in item or SORT_KEY not in item:
        msg = f"An item must carry its own '{PARTITION_KEY}' and '{SORT_KEY}'."
        raise ValueError(msg)
    params: dict[str, Any] = {
        "TableName": _table(),
        "Item": encode_item({SCHEMA_ATTRIBUTE: SCHEMA_VERSION} | item),
    }
    if condition is not None:
        params["ConditionExpression"] = condition
        if condition_values:
            params["ExpressionAttributeValues"] = encode_item(condition_values)
        if condition_names:
            params["ExpressionAttributeNames"] = dict(condition_names)
    try:
        await _client().put_item(**params)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == _CONDITION_FAILED:
            return False
        raise _failure(error, "PutItem") from error
    except BotoCoreError as error:
        raise _failure(error, "PutItem") from error
    return True


async def delete_item(
    partition_key: str,
    sort_key: str,
    *,
    condition: str | None = None,
    condition_values: Item | None = None,
) -> bool:
    """Delete one item, optionally only if a condition holds.

    Args:
        partition_key: The item's ``pk``.
        sort_key: The item's ``sk``.
        condition: A DynamoDB condition expression the delete requires, e.g.
            ``"holder = :me"`` to release only a lease still held.
        condition_values: Values the condition's ``:`` placeholders stand for.

    Returns:
        True when the delete was applied, False when the condition did not
        hold. Deleting an item that does not exist succeeds.

    Raises:
        TableUnavailableError: The delete did not complete.
    """
    params: dict[str, Any] = {
        "TableName": _table(),
        "Key": encode_item({PARTITION_KEY: partition_key, SORT_KEY: sort_key}),
    }
    if condition is not None:
        params["ConditionExpression"] = condition
        if condition_values:
            params["ExpressionAttributeValues"] = encode_item(condition_values)
    try:
        await _client().delete_item(**params)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == _CONDITION_FAILED:
            return False
        raise _failure(error, "DeleteItem") from error
    except BotoCoreError as error:
        raise _failure(error, "DeleteItem") from error
    return True


async def query_partition(
    partition_key: str,
    *,
    sort_key_prefix: str | None = None,
    consistent: bool = False,
    limit: int | None = None,
) -> list[Item]:
    """Read a partition, in sort-key order.

    Args:
        partition_key: The partition's ``pk``.
        sort_key_prefix: Keep only the items whose ``sk`` starts with this,
            which is how one kind of record is read out of a partition holding
            several.
        consistent: Read the latest writes rather than eventually consistent
            copies, at twice the read cost.
        limit: Stop after this many items.

    Returns:
        The decoded items, in ascending ``sk`` order, empty when the partition
        holds none.

    Raises:
        TableUnavailableError: The query did not complete.
    """
    values: Item = {":pk": partition_key}
    expression = f"{PARTITION_KEY} = :pk"
    if sort_key_prefix is not None:
        expression += f" AND begins_with({SORT_KEY}, :sk)"
        values[":sk"] = sort_key_prefix
    params: dict[str, Any] = {
        "TableName": _table(),
        "KeyConditionExpression": expression,
        "ExpressionAttributeValues": encode_item(values),
        "ConsistentRead": consistent,
    }
    if limit is not None:
        params["Limit"] = limit
    client = _client()
    items: list[Item] = []
    while True:
        try:
            response = await client.query(**params)
        except (BotoCoreError, ClientError) as error:
            raise _failure(error, "Query") from error
        try:
            items.extend(decode_item(item) for item in response.get("Items", ()))
        except ValueError as error:
            raise _decode_failure(error, "Query") from error
        # A page is bounded by 1 MB of read items, so a partition larger than
        # that answers in several -- silently truncated without this loop.
        last = response.get("LastEvaluatedKey")
        if not last or (limit is not None and len(items) >= limit):
            return items[:limit] if limit is not None else items
        params["ExclusiveStartKey"] = last


async def verify_table(start_event: EventLog) -> None:
    """Check the configured table at startup, so a broken one is known there.

    Reported and never fatal, matching every other operator-provided resource:
    a server refusing to boot would turn a table a moment away from existing,
    or an IAM policy still propagating, into an outage -- while every feature
    that needs the table still fails on its own terms until it is reachable.

    Args:
        start_event: Startup log event the findings are reported on.
    """
    if not (table := _table()):
        return
    try:
        client = _client()
        try:
            description = (await client.describe_table(TableName=table))["Table"]
        except (BotoCoreError, ClientError) as error:
            raise _failure(error, "DescribeTable") from error
        # Called after, not alongside: whatever denies or fails the first call
        # denies this one too, and one warning naming one IAM action is what
        # the operator acts on.
        try:
            ttl = (await client.describe_time_to_live(TableName=table))[
                "TimeToLiveDescription"
            ]
        except (BotoCoreError, ClientError) as error:
            raise _failure(error, "DescribeTimeToLive") from error
    except TableUnavailableError as unavailable:
        add_server_warning(start_event, unavailable.detail)
        return
    keys = {key["KeyType"]: key["AttributeName"] for key in description["KeySchema"]}
    if keys.get("HASH") != PARTITION_KEY or keys.get("RANGE") != SORT_KEY:
        add_server_warning(
            start_event,
            f"DynamoDB table '{table}' has the primary key {keys}, but the "
            f"features sharing it address items by '{PARTITION_KEY}' (partition "
            f"key) and '{SORT_KEY}' (sort key): recreate it with that key "
            "schema, since a sort key cannot be added to an existing table.",
        )
    if (
        ttl.get("TimeToLiveStatus") != _TTL_ENABLED
        or ttl.get("AttributeName") != EXPIRES_AT_ATTRIBUTE
    ):
        add_server_warning(
            start_event,
            f"DynamoDB table '{table}' has no time-to-live enabled on "
            f"'{EXPIRES_AT_ATTRIBUTE}': records meant to expire are kept "
            "forever. Enable it on that attribute.",
        )
