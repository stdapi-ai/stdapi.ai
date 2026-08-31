"""Tests for the DynamoDB table every shared-record feature is built on.

The table itself carries no feature: it is the marshalling, the key space, the
conditional writes and the degradation contract that the tenant-key and shared
model-cache features are assembled from. Everything here therefore asserts that
contract directly.

Two lanes, deliberately: the offline lane runs against a local DynamoDB stand-in
so the module stays covered without AWS credentials, and the sandbox lane runs
the same conditional-write semantics against the real service, which is the one
thing a stand-in can get subtly wrong.

Ref: stdapi/aws_dynamodb.py
     https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.CoreComponents.html
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from stdapi.aws_dynamodb import (
    EXPIRES_AT_ATTRIBUTE,
    KEY_SEPARATOR,
    PARTITION_KEY,
    SCHEMA_ATTRIBUTE,
    SCHEMA_VERSION,
    SORT_KEY,
    TABLE_REGION,
    TableUnavailableError,
    _failure,
    decode_item,
    decode_value,
    delete_item,
    encode_item,
    encode_value,
    get_item,
    item_key,
    put_item,
    query_partition,
    readable_schema,
    table_client_specs,
    verify_table,
)
from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from stdapi.aws_dynamodb import Item, ItemValue
    from stdapi.monitoring import EventLog

#: The lease condition every shared-record feature coordinates a refresh with.
_LEASE_CONDITION = (
    f"attribute_not_exists({PARTITION_KEY}) OR {EXPIRES_AT_ATTRIBUTE} < :now"
)


def _client_error(
    code: str, operation: str = "GetItem", message: str = "denied"
) -> ClientError:
    """Build the botocore error AWS answers a refused or impossible call with.

    Args:
        code: The AWS error code.
        operation: The operation that failed.
        message: The message AWS words the failure with; a denial IAM itself
            evaluated names the principal and the action in it.

    Returns:
        The error, shaped as botocore raises it.
    """
    return ClientError({"Error": {"Code": code, "Message": message}}, operation)


class TestKeys:
    """The composite key space every feature shares."""

    def test_a_key_joins_its_parts_with_the_reserved_separator(self) -> None:
        """A key is its parts, separated by the one character keys reserve.

        Ref: stdapi/aws_dynamodb.py:item_key
        """
        assert item_key("MODELCACHE", "abc") == f"MODELCACHE{KEY_SEPARATOR}abc"
        assert item_key("shard", "v1", "0") == "shard#v1#0"
        assert item_key("manifest") == "manifest"

    @pytest.mark.parametrize("part", ["a#b", f"x{KEY_SEPARATOR}"])
    def test_a_key_part_carrying_the_separator_is_refused(self, part: str) -> None:
        """A part that carries the separator could impersonate another namespace.

        ``item_key("KEY", "x#y")`` and ``item_key("KEY#x", "y")`` would otherwise
        be the same key, which is how one feature's records end up answering for
        another's.

        Ref: stdapi/aws_dynamodb.py:item_key
        """
        with pytest.raises(ValueError, match=re.escape(KEY_SEPARATOR)):
            item_key("KEY", part)

    def test_an_empty_key_part_is_refused(self) -> None:
        """An empty part collapses two different items onto one key.

        Ref: stdapi/aws_dynamodb.py:item_key
        """
        with pytest.raises(ValueError, match="must not be empty"):
            item_key("KEY", "")

    def test_a_key_needs_at_least_one_part(self) -> None:
        """An empty key is not a key.

        Ref: stdapi/aws_dynamodb.py:item_key
        """
        with pytest.raises(ValueError, match="at least one part"):
            item_key()

    def test_the_documented_key_spaces_are_the_implemented_ones(self) -> None:
        """The module docstring names every partition, and only real ones.

        The docstring is the operator-facing map of what the shared table
        holds; a namespace documented there but written by no code sends
        whoever audits the table looking for records that do not exist.

        Ref: stdapi/aws_dynamodb.py
        """
        import stdapi.aws_dynamodb as module  # noqa: PLC0415
        from stdapi.models._shared_cache import NAMESPACE  # noqa: PLC0415
        from stdapi.tenant_keys import _PARTITION  # noqa: PLC0415

        assert module.__doc__ is not None
        documented = {
            match.split(KEY_SEPARATOR, 1)[0]
            for match in re.findall(r"``pk=([^`]+)``", module.__doc__)
        }
        assert documented == {_PARTITION, NAMESPACE}


class TestSchemaVersion:
    """The version every item carries, so a rolling deployment stays readable."""

    def test_an_item_without_a_version_reads_as_this_build_s(self) -> None:
        """A stamped-by-default layout means an unstamped item is this one.

        Ref: stdapi/aws_dynamodb.py:readable_schema
        """
        assert readable_schema({})

    def test_an_older_or_equal_layout_is_readable(self) -> None:
        """This build understands the layout it writes, and everything before it.

        Ref: stdapi/aws_dynamodb.py:readable_schema
        """
        assert readable_schema({SCHEMA_ATTRIBUTE: SCHEMA_VERSION})

    def test_a_newer_layout_is_not_readable(self) -> None:
        """An item a newer build wrote must not be guessed at.

        During a rolling deployment both builds read the same table; what to do
        about an unreadable item is each feature's own decision, but reading it
        as if it were this layout is never one of them.

        Ref: stdapi/aws_dynamodb.py:readable_schema
        """
        assert not readable_schema({SCHEMA_ATTRIBUTE: SCHEMA_VERSION + 1})


class TestMarshalling:
    """The attribute-value subset this table stores."""

    @pytest.mark.parametrize(
        "value",
        [
            "text",
            "",
            0,
            -17,
            2**63,
            1.5,
            True,
            False,
            b"\x00\xff",
            None,
            [1, "two", None, [3]],
            {"a": {"b": [True]}},
            {"one", "two"},
        ],
    )
    def test_every_supported_value_round_trips(self, value: ItemValue) -> None:
        """Encoding then decoding a value returns the value.

        Ref: stdapi/aws_dynamodb.py:encode_value
             https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_AttributeValue.html
        """
        assert decode_value(encode_value(value)) == value

    def test_a_bool_is_not_encoded_as_a_number(self) -> None:
        """``bool`` is a subclass of ``int``; the order of the checks is the fix.

        Ref: stdapi/aws_dynamodb.py:encode_value
        """
        assert encode_value(True) == {"BOOL": True}  # noqa: FBT003 - the bool is the value under test

    def test_a_whole_item_round_trips(self) -> None:
        """Every attribute of an item survives the pair of conversions.

        Ref: stdapi/aws_dynamodb.py:encode_item
        """
        item: Item = {PARTITION_KEY: "KEY#a", SORT_KEY: "key", "disabled": False}

        assert decode_item(encode_item(item)) == item

    def test_a_fractional_number_stays_a_float(self) -> None:
        """DynamoDB stores every number as a string; the type comes from its shape.

        Ref: stdapi/aws_dynamodb.py:decode_value
        """
        assert decode_value({"N": "1.5"}) == 1.5
        assert isinstance(decode_value({"N": "17"}), int)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_number_dynamodb_cannot_hold_is_refused(self, value: float) -> None:
        """NaN and infinity have no DynamoDB representation.

        Encoding one would be accepted here and refused on the wire, where the
        failure is an opaque validation error instead of this message.

        Ref: stdapi/aws_dynamodb.py:encode_value
        """
        with pytest.raises(ValueError, match="NaN"):
            encode_value(value)

    def test_an_empty_set_is_refused(self) -> None:
        """DynamoDB has no empty set, and silently dropping the attribute lies.

        Ref: stdapi/aws_dynamodb.py:encode_value
        """
        with pytest.raises(ValueError, match="empty set"):
            encode_value(set())

    def test_a_type_this_table_does_not_store_is_refused(self) -> None:
        """An unsupported type fails here rather than as a wire error.

        Ref: stdapi/aws_dynamodb.py:encode_value
        """
        with pytest.raises(TypeError, match="complex"):
            encode_value(complex(1, 2))  # type: ignore[arg-type]

    def test_an_attribute_type_this_table_does_not_store_is_refused(self) -> None:
        """A number set written by something else is reported, never guessed at.

        Ref: stdapi/aws_dynamodb.py:decode_value
        """
        with pytest.raises(ValueError, match="'NS'"):
            decode_value({"NS": ["1"]})


class TestNothingIsConfigured:
    """A deployment that names no table opens nothing and calls nothing."""

    def test_no_client_is_pooled_without_a_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No table configured means no ``dynamodb`` client in the startup pool.

        This is what makes the feature cost nothing: with no client there is no
        connection, no call and no request to be billed for, and the Terraform
        module creates no table either.

        Ref: stdapi/aws_dynamodb.py:table_client_specs
             stdapi/main.py:lifespan
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", None)

        assert table_client_specs() == ()

    def test_the_client_is_pooled_once_a_table_is_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming a table adds exactly one client, in one region, with no failover.

        Ref: stdapi/aws_dynamodb.py:table_client_specs
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "stdapi-ai")
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_region", None)

        assert table_client_specs() == (("dynamodb", None),)

    async def test_a_read_without_a_table_names_the_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A feature calling in unconfigured is told which setting is missing.

        Ref: stdapi/aws_dynamodb.py:_client
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", None)

        with pytest.raises(TableUnavailableError, match="aws_dynamodb_table") as raised:
            await get_item("KEY#a", "key")

        assert "aws_dynamodb_table" in raised.value.detail

    async def test_a_startup_check_without_a_table_calls_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup makes no DynamoDB call at all when no table is configured.

        Ref: stdapi/aws_dynamodb.py:verify_table
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", None)
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert "server_warnings" not in start_event
        assert start_event["level"] == "info"


class TestDegradationContract:
    """What a failed call tells the operator, and never tells a caller."""

    def test_a_denial_names_the_action_to_grant(self) -> None:
        """The operator reads the exact IAM action their role is missing.

        Rendered by the same ``iam_denial_detail`` every other AWS denial the
        server reports goes through, so the action, the resource AWS named and
        the wording match the rest of the startup log rather than diverging as
        a copy of the shared rule would.

        Ref: stdapi/aws_dynamodb.py:_failure
             stdapi/api_errors.py:iam_denial_detail
             https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazondynamodb.html
        """
        error = _client_error(
            "AccessDeniedException",
            message=(
                "User: arn:aws:sts::123456789012:assumed-role/stdapi/task is "
                "not authorized to perform: dynamodb:GetItem on resource: "
                "arn:aws:dynamodb:us-east-1:123456789012:table/stdapi"
            ),
        )

        detail = _failure(error, "GetItem").detail

        assert "missing the IAM permission dynamodb:GetItem" in detail
        assert "arn:aws:dynamodb:us-east-1:123456789012:table/stdapi" in detail
        assert "until it is granted" in detail

    def test_an_invalid_credential_is_not_reported_as_a_policy_gap(self) -> None:
        """A rejected security token must not send the operator to IAM.

        DynamoDB answers ``UnrecognizedClientException`` when the credential is
        invalid, expired or the region is not enabled for the account -- none
        of which a policy change fixes. Every other caller in the server treats
        that code as a credential failure, so the shared renderer answers
        nothing for it and the generic branch names the code instead.

        Ref: stdapi/aws_dynamodb.py:_failure
             stdapi/api_errors.py:ACCESS_DENIED_CODES
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Programming.Errors.html
        """
        detail = _failure(
            _client_error("UnrecognizedClientException"), "DescribeTable"
        ).detail

        assert "UnrecognizedClientException" in detail
        assert "IAM permission" not in detail
        assert "granted" not in detail

    def test_a_missing_table_names_the_settings_to_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A table that is not there points at the two settings that name it.

        Ref: stdapi/aws_dynamodb.py:_failure
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "absent")

        detail = _failure(_client_error("ResourceNotFoundException"), "Query").detail

        assert "aws_dynamodb_table" in detail
        assert "aws_dynamodb_region" in detail

    async def test_a_call_with_no_pooled_client_names_the_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured table with no open client blames the pool, not the setting.

        The pool is built once at startup and cleared when any client fails to
        open, so a table configured with no client is a deployment that never
        finished starting or one already shutting down. Settings are immutable
        after startup, so the one thing this cannot be is the table name
        changing under a running server.

        Ref: stdapi/aws_dynamodb.py:_client
             stdapi/aws.py:get_client
        """
        from stdapi.aws import _CLIENTS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "configured")
        monkeypatch.delitem(_CLIENTS, "dynamodb", raising=False)

        with pytest.raises(TableUnavailableError) as raised:
            await get_item("KEY#a", "key")

        assert "client pool" in raised.value.detail
        assert TABLE_REGION in raised.value.detail

    def test_a_transport_failure_is_reported_without_the_aws_message(self) -> None:
        """Nothing AWS wrote is carried forward, not even into the operator log.

        Ref: stdapi/aws_dynamodb.py:_failure
             AGENTS.md "Never Leak Internals"
        """
        error = EndpointConnectionError(endpoint_url="https://dynamodb.example")

        detail = _failure(error, "PutItem").detail

        assert "EndpointConnectionError" in detail
        assert "dynamodb.example" not in detail


@pytest.mark.local
class TestTableAccess:
    """The read and write helpers, against a local DynamoDB stand-in."""

    async def test_an_item_round_trips(self, dynamodb_table: str) -> None:
        """What was written under a key is what is read back from it.

        Ref: stdapi/aws_dynamodb.py:put_item
        """
        item: Item = {
            PARTITION_KEY: item_key("KEY", "abc"),
            SORT_KEY: "key",
            "name": "a tenant",
            "disabled": False,
            "secret_hash": b"\x01\x02",
        }

        assert await put_item(item)

        assert await get_item(item_key("KEY", "abc"), "key") == item | {
            SCHEMA_ATTRIBUTE: SCHEMA_VERSION
        }

    async def test_a_missing_item_reads_as_nothing(self, dynamodb_table: str) -> None:
        """An absent key is None, never an error a feature has to classify.

        Ref: stdapi/aws_dynamodb.py:get_item
        """
        assert await get_item(item_key("KEY", "absent"), "key") is None

    async def test_a_written_item_carries_the_schema_version(
        self, dynamodb_table: str
    ) -> None:
        """Every item is stamped, so no feature can forget to stamp its own.

        Ref: stdapi/aws_dynamodb.py:put_item
        """
        await put_item({PARTITION_KEY: "KEY#v", SORT_KEY: "key"})

        item = await get_item("KEY#v", "key")

        assert item is not None
        assert item[SCHEMA_ATTRIBUTE] == SCHEMA_VERSION

    async def test_a_newer_schema_item_round_trips_intact(
        self, dynamodb_table: str
    ) -> None:
        """A newer build's item still decodes whole; only interpreting it is refused.

        During a rolling deployment, the old build must still read the item
        back byte for byte -- it is :func:`readable_schema`, applied by the
        caller, that decides not to trust it.

        Ref: stdapi/aws_dynamodb.py:readable_schema
        """
        await put_item(
            {
                PARTITION_KEY: "KEY#newer",
                SORT_KEY: "key",
                SCHEMA_ATTRIBUTE: SCHEMA_VERSION + 1,
                "future_field": "unfamiliar but supported",
            }
        )

        item = await get_item("KEY#newer", "key")

        assert item is not None
        assert item["future_field"] == "unfamiliar but supported"
        assert not readable_schema(item)

    async def test_an_item_this_build_cannot_decode_degrades_like_any_other_failure(
        self, dynamodb_table: str
    ) -> None:
        """A type this table never writes still degrades, rather than raising raw.

        Every feature sharing this table catches only :class:`TableUnavailableError`;
        an item written by something else, carrying a type ``decode_value`` does
        not support, must not escape as a bare ``ValueError``.

        Ref: stdapi/aws_dynamodb.py:get_item
        """
        from stdapi.aws import get_client  # noqa: PLC0415

        await get_client("dynamodb").put_item(
            TableName=dynamodb_table,
            Item={
                PARTITION_KEY: {"S": "KEY#weird"},
                SORT_KEY: {"S": "key"},
                "odd": {"NS": ["1", "2"]},
            },
        )

        with pytest.raises(TableUnavailableError, match="cannot decode"):
            await get_item("KEY#weird", "key")

    async def test_an_item_must_carry_its_own_key(self, dynamodb_table: str) -> None:
        """An item without both key attributes is refused before the call.

        Ref: stdapi/aws_dynamodb.py:put_item
        """
        with pytest.raises(ValueError, match=SORT_KEY):
            await put_item({PARTITION_KEY: "KEY#a"})

    async def test_a_consistent_read_sees_the_last_write(
        self, dynamodb_table: str
    ) -> None:
        """The strongly consistent read is available where a feature needs it.

        Ref: stdapi/aws_dynamodb.py:get_item
        """
        await put_item({PARTITION_KEY: "KEY#c", SORT_KEY: "key", "n": 1})

        item = await get_item("KEY#c", "key", consistent=True)

        assert item is not None
        assert item["n"] == 1

    async def test_only_one_writer_takes_a_lease(self, dynamodb_table: str) -> None:
        """A conditional write is how a fleet elects one refresher and no more.

        The losing write is not an error: it returns False, which is exactly the
        signal a caller needs to serve what it already has instead.

        Ref: stdapi/aws_dynamodb.py:put_item
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html
        """
        lease: Item = {
            PARTITION_KEY: item_key("MODELCACHE", "fp"),
            SORT_KEY: "lease",
            EXPIRES_AT_ATTRIBUTE: 2_000_000_000,
            "holder": "first",
        }

        won = await put_item(
            lease, condition=_LEASE_CONDITION, condition_values={":now": 1_000_000_000}
        )
        lost = await put_item(
            lease | {"holder": "second"},
            condition=_LEASE_CONDITION,
            condition_values={":now": 1_000_000_000},
        )

        assert won
        assert not lost
        held = await get_item(item_key("MODELCACHE", "fp"), "lease")
        assert held is not None
        assert held["holder"] == "first"

    async def test_an_expired_lease_is_taken_over(self, dynamodb_table: str) -> None:
        """A writer that died holding the lease does not wedge the fleet.

        Ref: stdapi/aws_dynamodb.py:put_item
        """
        lease: Item = {
            PARTITION_KEY: item_key("MODELCACHE", "expired"),
            SORT_KEY: "lease",
            EXPIRES_AT_ATTRIBUTE: 1_000,
            "holder": "crashed",
        }
        await put_item(lease)

        assert await put_item(
            lease | {"holder": "next", EXPIRES_AT_ATTRIBUTE: 2_000_000_000},
            condition=_LEASE_CONDITION,
            condition_values={":now": 1_000_000_000},
        )

    async def test_a_delete_can_require_the_item_to_still_be_ours(
        self, dynamodb_table: str
    ) -> None:
        """Releasing a lease someone else took over must not succeed.

        Ref: stdapi/aws_dynamodb.py:delete_item
        """
        await put_item(
            {PARTITION_KEY: "MODELCACHE#d", SORT_KEY: "lease", "holder": "other"}
        )

        released = await delete_item(
            "MODELCACHE#d",
            "lease",
            condition="holder = :me",
            condition_values={":me": "mine"},
        )

        assert not released
        assert await get_item("MODELCACHE#d", "lease") is not None

    async def test_deleting_what_is_not_there_succeeds(
        self, dynamodb_table: str
    ) -> None:
        """An unconditional delete is idempotent, so cleanup never has to check.

        Ref: stdapi/aws_dynamodb.py:delete_item
        """
        assert await delete_item("MODELCACHE#gone", "lease")

    async def test_a_partition_reads_in_sort_key_order(
        self, dynamodb_table: str
    ) -> None:
        """A partition comes back ordered, which is what makes shards reassemble.

        Ref: stdapi/aws_dynamodb.py:query_partition
        """
        partition = item_key("MODELCACHE", "ordered")
        for index in (2, 0, 1):
            await put_item(
                {
                    PARTITION_KEY: partition,
                    SORT_KEY: item_key("shard", "v1", str(index)),
                }
            )

        items = await query_partition(partition)

        assert [item[SORT_KEY] for item in items] == [
            "shard#v1#0",
            "shard#v1#1",
            "shard#v1#2",
        ]

    async def test_a_prefix_reads_one_kind_out_of_a_partition(
        self, dynamodb_table: str
    ) -> None:
        """One partition holds several kinds; a prefix reads exactly one of them.

        This is the reason the sort key is ``<kind>#<discriminator>`` for a kind
        with several records: the manifest and the lease sharing the partition
        must not come back with the shards.

        Ref: stdapi/aws_dynamodb.py:query_partition
        """
        partition = item_key("MODELCACHE", "mixed")
        for sort_key in ("manifest", "lease", "shard#v1#0", "shard#v2#0"):
            await put_item({PARTITION_KEY: partition, SORT_KEY: sort_key})

        items = await query_partition(partition, sort_key_prefix="shard#v1#")

        assert [item[SORT_KEY] for item in items] == ["shard#v1#0"]

    async def test_a_query_stops_at_its_limit(self, dynamodb_table: str) -> None:
        """A bounded query reads no more than it was asked for.

        Ref: stdapi/aws_dynamodb.py:query_partition
        """
        partition = item_key("MODELCACHE", "limited")
        for index in range(3):
            await put_item({PARTITION_KEY: partition, SORT_KEY: f"shard#v1#{index}"})

        assert len(await query_partition(partition, limit=2)) == 2

    async def test_a_query_degrades_on_an_item_it_cannot_decode(
        self, dynamodb_table: str
    ) -> None:
        """A partition holding a foreign item degrades the whole read, not raises raw.

        Ref: stdapi/aws_dynamodb.py:query_partition
        """
        from stdapi.aws import get_client  # noqa: PLC0415

        partition = item_key("MODELCACHE", "weird")
        await get_client("dynamodb").put_item(
            TableName=dynamodb_table,
            Item={
                PARTITION_KEY: {"S": partition},
                SORT_KEY: {"S": "shard#v1#0"},
                "odd": {"NS": ["1"]},
            },
        )

        with pytest.raises(TableUnavailableError, match="cannot decode"):
            await query_partition(partition)

    async def test_an_empty_partition_reads_as_nothing(
        self, dynamodb_table: str
    ) -> None:
        """A partition nobody wrote to is empty, not an error.

        Ref: stdapi/aws_dynamodb.py:query_partition
        """
        assert await query_partition(item_key("MODELCACHE", "empty")) == []

    async def test_a_partition_past_one_page_reads_whole(
        self, dynamodb_table: str
    ) -> None:
        """A partition past DynamoDB's 1 MB page answers in full, not truncated.

        Four shards near the real model cache's own size cross the page
        boundary, which is what makes the ``ExclusiveStartKey`` continuation
        the query loop carries forward, rather than the ``limit`` short
        circuit, the branch under test.

        Ref: stdapi/aws_dynamodb.py:query_partition
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/QueryAndScan.html#Query.Pagination
        """
        from secrets import token_bytes  # noqa: PLC0415

        partition = item_key("MODELCACHE", "paginated")
        for index in range(4):
            await put_item(
                {
                    PARTITION_KEY: partition,
                    SORT_KEY: item_key("shard", "v1", str(index)),
                    "data": token_bytes(350_000),
                }
            )

        items = await query_partition(partition)

        assert [item[SORT_KEY] for item in items] == [
            "shard#v1#0",
            "shard#v1#1",
            "shard#v1#2",
            "shard#v1#3",
        ]

    @pytest.mark.parametrize(
        ("operation", "call"),
        [
            ("GetItem", lambda: get_item("KEY#a", "key")),
            ("PutItem", lambda: put_item({PARTITION_KEY: "KEY#a", SORT_KEY: "key"})),
            ("DeleteItem", lambda: delete_item("KEY#a", "key")),
            ("Query", lambda: query_partition("KEY#a")),
        ],
    )
    async def test_every_call_degrades_on_a_service_error(
        self,
        dynamodb_table: str,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
        call: Callable[[], Awaitable[object]],
    ) -> None:
        """A table that answers with an error degrades whichever helper asked.

        Every feature sharing this table catches only
        :class:`TableUnavailableError`, so a helper letting a raw ``ClientError``
        through -- on a throttle, or on a table deleted under a running server --
        would reach a caller as an unhandled exception.

        Ref: stdapi/aws_dynamodb.py:_failure
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "deleted-under-us")

        with pytest.raises(TableUnavailableError) as raised:
            await call()

        assert f"dynamodb:{operation}" in raised.value.detail
        assert "aws_dynamodb_table" in raised.value.detail


@pytest.mark.local
class TestStartupCheck:
    """What startup reports about the table an operator pointed the server at."""

    async def test_a_healthy_table_is_reported_silently(
        self, dynamodb_table: str
    ) -> None:
        """A table with the right key schema and time-to-live warns about nothing.

        Ref: stdapi/aws_dynamodb.py:verify_table
        """
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert "server_warnings" not in start_event

    async def test_a_missing_table_is_reported_and_never_fatal(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A table that is not there warns the operator instead of stopping boot.

        A server refusing to start would turn a table a moment away from
        existing, or an IAM policy still propagating, into an outage.

        Ref: stdapi/aws_dynamodb.py:verify_table
        """
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", "not-created-yet")
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert start_event["level"] == "warning"
        assert "not-created-yet" in str(start_event["server_warnings"])

    async def test_a_table_without_time_to_live_is_reported(
        self, dynamodb_table: str
    ) -> None:
        """Records meant to expire would be kept forever, which the operator pays for.

        Ref: stdapi/aws_dynamodb.py:verify_table
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html
        """
        from stdapi.aws import get_client  # noqa: PLC0415

        await get_client("dynamodb").update_time_to_live(
            TableName=dynamodb_table,
            TimeToLiveSpecification={
                "Enabled": False,
                "AttributeName": EXPIRES_AT_ATTRIBUTE,
            },
        )
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert EXPIRES_AT_ATTRIBUTE in str(start_event["server_warnings"])

    async def test_time_to_live_on_another_attribute_is_reported(
        self, dynamodb_table: str
    ) -> None:
        """Enabled is not enough: it has to expire the attribute these features write.

        A table can carry only one time-to-live attribute, so one pointed
        somewhere else is indistinguishable from none at all -- and it is the
        likelier mistake, since enabling it at all is the deliberate step.

        Ref: stdapi/aws_dynamodb.py:verify_table
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html
        """
        from stdapi.aws import get_client  # noqa: PLC0415

        await get_client("dynamodb").update_time_to_live(
            TableName=dynamodb_table,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert EXPIRES_AT_ATTRIBUTE in str(start_event["server_warnings"])

    async def test_a_table_without_a_sort_key_is_reported(
        self, dynamodb_table: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one mistake that cannot be repaired in place is named at startup.

        A sort key cannot be added to an existing table, so an operator who
        created a partition-key-only table has to recreate it -- and has to be
        told that before a feature starts writing to it.

        Ref: stdapi/aws_dynamodb.py:verify_table
             https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_UpdateTable.html
        """
        from stdapi.aws import get_client  # noqa: PLC0415

        flat = "stdapi-test-flat"
        await get_client("dynamodb").create_table(
            TableName=flat,
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        monkeypatch.setattr(SETTINGS, "aws_dynamodb_table", flat)
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert "sort key" in str(start_event["server_warnings"])


@pytest.mark.gateway("Amazon DynamoDB has no upstream-vendor equivalent")
@pytest.mark.xdist_group("dynamodb")
class TestRealTable:
    """The semantics a stand-in can get subtly wrong, against the real service.

    Only what depends on DynamoDB's own behaviour rather than on this module's:
    the atomicity of a conditional write under a genuine race, the item size a
    shard has to fit inside, and that the startup check agrees with the schema
    the deployment module creates.

    These call the module in process, so they take ``sandbox_dynamodb``: it
    fills the AWS client pool ``_client`` reads with a client opened on the loop
    the test runs on. The app's own pool cannot serve them -- its lifespan runs
    inside ``TestClient``'s portal, on another loop -- and without the fixture
    every one of them raises :class:`TableUnavailableError` instead.

    The fixture's table is created here, so what an operator's *deployed* table
    carries is not settled by these; that is the deployment's own check.
    """

    async def test_the_startup_check_accepts_the_schema_the_module_creates(
        self, sandbox_dynamodb: str
    ) -> None:
        """A table shaped like the module's passes the check that gates startup.

        Ref: stdapi/aws_dynamodb.py:verify_table
        """
        del sandbox_dynamodb
        start_event: EventLog = {"type": "start", "level": "info"}  # type: ignore[typeddict-item]

        await verify_table(start_event)

        assert "server_warnings" not in start_event

    async def test_one_writer_wins_a_real_conditional_write(
        self, sandbox_dynamodb: str
    ) -> None:
        """DynamoDB itself, not the stand-in, settles the race for the lease.

        Ref: stdapi/aws_dynamodb.py:put_item
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html
        """
        from asyncio import gather  # noqa: PLC0415
        from secrets import token_hex  # noqa: PLC0415

        del sandbox_dynamodb
        partition = item_key("MODELCACHE", f"test-{token_hex(8)}")
        lease: dict[str, Any] = {
            PARTITION_KEY: partition,
            SORT_KEY: "lease",
            EXPIRES_AT_ATTRIBUTE: 2_000_000_000,
        }
        try:
            outcomes = await gather(
                *(
                    put_item(
                        lease | {"holder": str(index)},
                        condition=_LEASE_CONDITION,
                        condition_values={":now": 1_000_000_000},
                    )
                    for index in range(4)
                )
            )

            assert sum(outcomes) == 1
        finally:
            await delete_item(partition, "lease")

    async def test_a_shared_model_cache_shard_fits_in_one_item(
        self, sandbox_dynamodb: str
    ) -> None:
        """A full-size shard is accepted by the real service, not just the stand-in.

        The stand-in enforces no item size, so the one number the shared model
        cache cannot verify offline is that its shard size really does leave
        room for the key, the attribute names and the item overhead inside
        DynamoDB's 400 KB limit.

        Ref: stdapi/models/_shared_cache.py:_SHARD_BYTES
             https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ServiceQuotas.html
        """
        from secrets import token_bytes, token_hex  # noqa: PLC0415

        from stdapi.models._shared_cache import _SHARD_BYTES  # noqa: PLC0415

        del sandbox_dynamodb
        partition = item_key("MODELCACHE", f"test-{token_hex(8)}")
        sort_key = item_key("shard", token_hex(8), "0000")
        try:
            assert await put_item(
                {
                    PARTITION_KEY: partition,
                    SORT_KEY: sort_key,
                    "data": token_bytes(_SHARD_BYTES),
                    EXPIRES_AT_ATTRIBUTE: 2_000_000_000,
                }
            )
        finally:
            await delete_item(partition, sort_key)
