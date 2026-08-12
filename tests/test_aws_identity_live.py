"""Live AWS checks of the per-end-user role sessions.

``tests/test_aws_identity.py`` stubs AWS STS in every one of its tests, so the
whole feature is asserted there against a service that answers whatever the
stub says. Three of its contracts are IAM contracts, and only a real account
answers those:

- AWS STS validates ``RoleSessionName`` and the session tag value against its
  own character sets and lengths, and it does so *before* it authorizes the
  call. ``user_role_session_identity`` maps an arbitrary end user identifier
  into both, and nothing short of a real ``sts:AssumeRole`` shows the result is
  something AWS accepts. The mapping checks below therefore send the mapped
  values to AWS STS naming a role ARN that does not exist: a mapping AWS
  rejects fails with ``ValidationError``, a mapping it accepts gets past
  validation and fails with ``AccessDenied``. No IAM resource is involved and
  no session is ever opened, so those checks need only AWS credentials.
- ``sts:TagSession`` is authorized separately from ``sts:AssumeRole``: a trust
  policy granting only the second denies every tagged session, which is the
  failure a deployment meets on its first request and which no mock reports.
- The ``RoleSessionName`` AWS records is what reaches
  ``line_item_iam_principal`` in the Cost and Usage Report, so it is read back
  from AWS rather than trusted from the value the gateway computed.

The session tests need a role to assume. It is an IAM resource of the account
under test, created outside this repository -- see
``docs/operations_iam_permissions.md#per-user-cost-attribution`` for the trust
and permission policies it needs -- and named by ``TEST_BEDROCK_USER_ROLE_ARN``
in ``tests/.env``; a checkout without one skips.

Ref: https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-iam-principal-tracking.html
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/aws.py:user_role_credentials
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import stdapi.aws
from stdapi.aws import (
    clear_user_role_cache,
    user_role_credentials,
    user_role_session_identity,
)
from stdapi.config import AWS_REGION, AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

#: In-process implementation signed against real AWS; meaningless on a remote target.
#: Grouped because the role session cache and ``SETTINGS`` are process-global.
pytestmark = [pytest.mark.local, pytest.mark.xdist_group("aws_user_role_live")]

#: End user identifier the live sessions are opened for; no real user collides with it.
_IDENTITY = "stdapi-ai-live-test-user"

#: Session tag key the live sessions carry, independent of the deployment's own.
_TAG_KEY = "user"

#: Cheapest chat model, invoked once to prove the assumed session can call Bedrock.
_MODEL_ID = "amazon.nova-micro-v1:0"

#: Output tokens the live invocation is allowed; the answer itself is irrelevant.
_MAX_TOKENS = 5

#: End user identifiers AWS STS must accept once mapped, one per mapping rule.
_ACCEPTED_IDENTITIES = {
    "plain": "alice",
    "email-and-path": "alice@example.com/tenant-7",
    "unicode-letters": "Zoé Müller <zoe@example.com>",
    "no-ascii-word-characters": "田中太郎",
    "nothing-mappable": "!!! ***",
}

#: Identifiers AWS STS must reject unmapped, so the mapping above is load-bearing.
_REJECTED_IDENTITIES = {
    "path-separator": "alice@example.com/tenant-7",
    "over-64-characters": "a" * 65,
    "angle-brackets": "<alice>",
}


async def _assume_role_error(
    role_arn: str, session_name: str, tag_value: str
) -> ClientError:
    """Return the error AWS STS answers an ``AssumeRole`` that cannot succeed.

    Args:
        role_arn: Role to name; it does not exist, so the call always fails.
        session_name: RoleSessionName to submit.
        tag_value: Session tag value to submit.

    Returns:
        The ``ClientError`` AWS STS raised, whose code says whether the call
        was rejected as malformed or reached authorization.
    """
    async with AWS_SESSION.create_client("sts", region_name=AWS_REGION) as sts:
        with pytest.raises(ClientError) as raised:
            await sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName=session_name,
                Tags=[{"Key": _TAG_KEY, "Value": tag_value}],
                DurationSeconds=900,
            )
    return raised.value


@pytest.fixture(scope="module")
def absent_role_arn(aws_account_id: str) -> str:
    """A syntactically valid role ARN of the test account that does not exist.

    Naming a role that exists would make the outcome depend on that role's
    trust policy; naming one that does not keeps every mapping check an
    authorization failure, whatever the account contains.
    """
    return f"arn:aws:iam::{aws_account_id}:role/stdapi-ai-absent-live-test-role"


class TestSessionIdentityAcceptedByAws:
    """AWS STS accepts what ``user_role_session_identity`` produces, and only that.

    Every assertion here reads the code AWS STS answered: it validates the
    request before authorizing it, so ``AccessDenied`` means the submitted
    session name and tag value satisfied the service's own constraints, and
    ``ValidationError`` means they did not.

    Ref: https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
         stdapi/aws.py:user_role_session_identity
    """

    @pytest.mark.parametrize(
        "identity", _ACCEPTED_IDENTITIES.values(), ids=_ACCEPTED_IDENTITIES.keys()
    )
    async def test_mapped_identity_is_valid_to_aws(
        self, absent_role_arn: str, identity: str
    ) -> None:
        """A mapped end user identifier passes AWS STS request validation.

        The identifiers cover each rule the mapping applies: characters legal
        in a tag value but not in a session name, characters legal in neither,
        and an identifier with nothing AWS accepts left in it at all.
        """
        session_name, tag_value = user_role_session_identity(identity)
        error = await _assume_role_error(absent_role_arn, session_name, tag_value)
        assert error.response["Error"]["Code"] == "AccessDenied", (
            f"AWS STS rejected the mapping of {identity!r} "
            f"({session_name!r}, {tag_value!r}): "
            f"{error.response['Error']['Message']}"
        )

    @pytest.mark.parametrize(
        "identity", _REJECTED_IDENTITIES.values(), ids=_REJECTED_IDENTITIES.keys()
    )
    async def test_unmapped_identity_is_invalid_to_aws(
        self, absent_role_arn: str, identity: str
    ) -> None:
        """An end user identifier AWS STS rejects verbatim is rejected verbatim.

        Without this the check above proves nothing: an ``AccessDenied`` would
        be just as reachable if AWS STS validated nothing at all.
        """
        error = await _assume_role_error(absent_role_arn, identity, identity)
        assert error.response["Error"]["Code"] == "ValidationError", (
            f"AWS STS accepted the unmapped identifier {identity!r}, so the "
            f"mapping is not what makes the session name valid"
        )

    async def test_long_identity_is_truncated_to_what_aws_accepts(
        self, absent_role_arn: str
    ) -> None:
        """An identifier past both limits is mapped onto the limits exactly.

        AWS STS caps a role session name at 64 characters and a session tag
        value at 256, and the mapping spends the last 13 of each on the digest
        keeping two end users apart -- so an over-long identifier must land on
        the limit, not one past it.
        """
        session_name, tag_value = user_role_session_identity("u" * 300)
        assert len(session_name) == 64
        assert len(tag_value) == 256
        error = await _assume_role_error(absent_role_arn, session_name, tag_value)
        assert error.response["Error"]["Code"] == "AccessDenied", (
            f"AWS STS rejected the truncated session: "
            f"{error.response['Error']['Message']}"
        )


class TestEndUserRoleSession:
    """A real role session opened for an end user, and a model call signed with it.

    This is the part no mock reaches: the role's trust policy has to grant this
    server both ``sts:AssumeRole`` and ``sts:TagSession``, and the role's own
    permission policy has to allow the invocation actions on every model ARN
    form the request can resolve to.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-iam-principal-tracking.html
         stdapi/aws.py:user_role_credentials
    """

    @pytest.fixture
    async def user_role(
        self, bedrock_user_role_arn: str, monkeypatch: pytest.MonkeyPatch
    ) -> AsyncIterator[str]:
        """Enable per-end-user attribution against the configured role.

        The AWS STS client the feature reads from the pool is created here
        rather than by entering the server's connection manager, whose teardown
        empties the process-wide pool a live target may still be serving from.

        Yields:
            The role ARN sessions are opened on.
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_user_role_arn", bedrock_user_role_arn
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_tag_key", _TAG_KEY)
        async with AWS_SESSION.create_client("sts", region_name=AWS_REGION) as sts:
            monkeypatch.setitem(stdapi.aws._CLIENTS, "sts", {AWS_REGION: sts})  # noqa: SLF001
            clear_user_role_cache()
            yield bedrock_user_role_arn
            clear_user_role_cache()

    async def test_aws_records_the_session_name_the_gateway_computed(
        self, user_role: str
    ) -> None:
        """AWS reports the session under the name the gateway mapped the user to.

        Read back from the assumed session's own caller identity: that name is
        what AWS writes to ``line_item_iam_principal``, so a mapping the gateway
        computed but AWS stored differently would attribute the spend to
        something no report can be grouped by.
        """
        session_name, _ = user_role_session_identity(_IDENTITY)
        credentials = await user_role_credentials(_IDENTITY)
        frozen = await credentials.get_frozen_credentials()
        async with AWS_SESSION.create_client(
            "sts",
            region_name=AWS_REGION,
            aws_access_key_id=frozen.access_key,
            aws_secret_access_key=frozen.secret_key,
            aws_session_token=frozen.token,
        ) as sts:
            identity = await sts.get_caller_identity()
        role_name = user_role.rpartition("/")[2]
        assert identity["Arn"].endswith(f":assumed-role/{role_name}/{session_name}"), (
            f"AWS recorded {identity['Arn']}, not the session name the gateway "
            f"computed ({session_name})"
        )
        assert credentials.session_name == session_name

    async def test_truncated_identity_opens_a_session_aws_accepts(
        self, user_role: str
    ) -> None:
        """An over-long end user identifier still opens a session AWS grants.

        The sanitisation is exercised end to end here rather than against
        request validation alone: the session it produces is one AWS issues
        credentials for.
        """
        session_name, _ = user_role_session_identity(f"{_IDENTITY}-{'x' * 300}")
        assert len(session_name) == 64
        credentials = await user_role_credentials(f"{_IDENTITY}-{'x' * 300}")
        frozen = await credentials.get_frozen_credentials()
        assert frozen.access_key
        assert frozen.token
        assert credentials.session_name == session_name

    async def test_model_invocation_signed_as_the_end_user_succeeds(
        self, user_role: str, request_log: dict[str, Any]
    ) -> None:
        """A Bedrock invocation signed with the end user's session is authorized.

        The assertion that no mock can make: it fails when the role's trust
        policy omits ``sts:TagSession``, and when the role's permission policy
        misses an ARN form the request resolves to. The client is built from the
        server's own session, so the invocation is substituted by the very hook
        a request goes through.
        """
        request_log["request_user_id"] = _IDENTITY
        session_name, _ = user_role_session_identity(_IDENTITY)
        async with AWS_SESSION.create_client(
            "bedrock-runtime", region_name=SETTINGS.aws_bedrock_regions[0]
        ) as client:
            response = await client.converse(
                modelId=_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
                inferenceConfig={"maxTokens": _MAX_TOKENS, "temperature": 0.0},
            )
        assert response["usage"]["outputTokens"] >= 1
        assert request_log["aws_role_session_name"] == session_name
