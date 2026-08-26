"""Tests for the per-tenant cross-account AWS credentials (#154).

A tenant may register an IAM role of its own AWS account against its API key;
the gateway assumes it with a server-minted ExternalId and signs that tenant's
model invocations with the session, moving quota and spend to the tenant's
account. The adversarial shapes matter more than the happy path: a wrong
ExternalId, a session that dies mid-flight, error bodies that must name none
of the gateway's own account details, a tenant throttle that must not poison
the shared region backoff, and every service the credential cannot pay for
(Mantle, Marketplace, batch, realtime, asynchronous generation) refusing
loudly instead of billing the operator.

Ref: stdapi/aws.py:tenant_role_credentials
     stdapi/api_errors.py:denied_feature_unavailable
     https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html
     https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_terms-and-concepts.html
"""

from __future__ import annotations

import re
from asyncio import gather, sleep
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from aiobotocore.config import AioConfig
from botocore.exceptions import ClientError, EndpointConnectionError
from pydantic import ValidationError
from starlette.requests import HTTPConnection

import stdapi.models
import stdapi.usage
from stdapi import aws
from stdapi.api_errors import (
    ApiError,
    TenantCredentialError,
    denied_feature_unavailable,
)
from stdapi.aws import (
    TENANT_ACCESS_DENIED_MESSAGE,
    TENANT_CREDENTIAL_FAILURE_MESSAGE,
    TENANT_REALTIME_MESSAGE,
    TENANT_SESSION_UNAVAILABLE_MESSAGE,
    request_signing_credentials,
    signed_as_tenant,
    tenant_role_credentials,
    verify_bidi_user_role_policy,
)
from stdapi.config import AWS_SESSION, SETTINGS, ModelAliasConfig, _Settings
from stdapi.models import (
    MANTLE_SERVICE,
    MARKETPLACE_SERVICE,
    ModelDetails,
    route_and_execute,
)
from stdapi.monitoring import (
    REQUEST_LOG,
    TENANT,
    Tenant,
    TenantAwsCredential,
    tenant_aws_credential,
)
from stdapi.region_routing import RegionRouter
from stdapi.utils import hide_security_details
from tests._helpers import make_event_log

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from botocore.awsrequest import AWSPreparedRequest

#: All tests in this module drive the in-process implementation and its globals.
pytestmark = pytest.mark.local

#: Key ID of the tenant every test registers its credential against.
_KEY_ID = "T" + "0" * 15

#: Access key the pooled client itself is built on (the server's own identity).
_SERVER_KEY = "AKIASERVERROLEXXXXXX"

#: Role the tests register; the account ID is the tenant's, never the gateway's.
_ROLE_ARN = "arn:aws:iam::210987654321:role/stdapi-tenant"

#: ExternalId the gateway minted for the tenant.
_EXTERNAL_ID = "external-id-under-test"

#: Account ID standing for the gateway's own in leaked-detail assertions.
_GATEWAY_ACCOUNT = "111122223333"

#: Matcher for anything shaped like an ARN or an AWS account ID.
_INTERNAL_DETAIL_RE = re.compile(r"arn:aws|\b\d{12}\b|Exception|assumed-role")


def _tenant(credential: TenantAwsCredential | None = None) -> Tenant:
    """Build a verified tenant carrying *credential*.

    Args:
        credential: The registered AWS credential, if any.

    Returns:
        The tenant.
    """
    return Tenant(key_id=_KEY_ID, name="tenant", aws_credential=credential)


@pytest.fixture
def tenant_signing(monkeypatch: pytest.MonkeyPatch) -> Iterator[Tenant]:
    """Install a request context whose tenant registered an AWS credential.

    Yields:
        The verified tenant.
    """
    monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
    tenant = _tenant(TenantAwsCredential(role_arn=_ROLE_ARN, external_id=_EXTERNAL_ID))
    token = TENANT.set(tenant)
    aws.clear_tenant_role_cache()
    try:
        yield tenant
    finally:
        TENANT.reset(token)
        aws.clear_tenant_role_cache()


@pytest.fixture
def fake_sts(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Serve AssumeRole from a local stub, recording every call.

    Returns:
        A namespace exposing ``calls`` (recorded kwargs), ``failure`` (an
        exception to raise instead of answering, when set) and ``expires_in``
        (seconds until the served session expires -- an hour by default, far
        enough away that no refresh triggers).
    """
    state = SimpleNamespace(calls=[], failure=None, expires_in=3600.0)

    async def assume_role(**kwargs: object) -> dict[str, Any]:
        state.calls.append(kwargs)
        # Suspends like the real call does, so concurrent openers interleave.
        await sleep(0)
        if state.failure is not None:
            raise state.failure
        return {
            "Credentials": {
                "AccessKeyId": "AKIDTEST",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
                "Expiration": datetime.now(UTC) + timedelta(seconds=state.expires_in),
            }
        }

    monkeypatch.setitem(
        aws._CLIENTS,  # noqa: SLF001
        "sts",
        {SETTINGS.aws_bedrock_regions[0]: SimpleNamespace(assume_role=assume_role)},
    )
    return state


class TestConfiguration:
    """The combinations the settings model must refuse at startup.

    Ref: stdapi/config.py:_validate_tenant_credentials
    """

    def test_the_feature_requires_tenant_api_keys(self) -> None:
        """Enabling credentials without tenant keys fails startup."""
        with pytest.raises(ValidationError, match="requires tenant_api_keys"):
            _Settings(tenant_aws_credentials=True)

    def test_a_gateway_guardrail_refuses_the_feature(self) -> None:
        """A configured guardrail cannot coexist with tenant credentials.

        Fail closed at configuration time: a tenant principal cannot evaluate
        a guardrail of the gateway's account, and dropping it silently would
        serve unfiltered content the operator believes is guarded.
        """
        with pytest.raises(ValidationError, match="incompatible with Amazon Bedrock"):
            _Settings(
                tenant_aws_credentials=True,
                tenant_api_keys=True,
                aws_dynamodb_table="shared",
                tenant_key_ssm_parameter_prefix="/test/tenant-keys",
                aws_bedrock_guardrail_identifier="gr-1",
                aws_bedrock_guardrail_version="1",
            )

    def test_an_alias_guardrail_refuses_the_feature(self) -> None:
        """A model alias carrying a guardrail is deployment configuration too."""
        with pytest.raises(ValidationError, match="incompatible with Amazon Bedrock"):
            _Settings(
                tenant_aws_credentials=True,
                tenant_api_keys=True,
                aws_dynamodb_table="shared",
                tenant_key_ssm_parameter_prefix="/test/tenant-keys",
                model_aliases={
                    "guarded": ModelAliasConfig(
                        model="amazon.nova-lite-v1:0",
                        guardrail_identifier="gr-1",
                        guardrail_version="1",
                    )
                },
            )

    def test_the_feature_composes_with_tenant_keys(self) -> None:
        """The valid combination configures without error."""
        settings = _Settings(
            tenant_aws_credentials=True,
            tenant_api_keys=True,
            aws_dynamodb_table="shared",
            tenant_key_ssm_parameter_prefix="/test/tenant-keys",
        )
        assert settings.tenant_aws_credentials


class TestRoleSession:
    """Opening, caching and failing the tenant's cross-account session.

    Ref: stdapi/aws.py:_assume_tenant_role
         https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
    """

    async def test_the_session_presents_the_minted_external_id(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """AssumeRole carries the role, the ExternalId and the 1 h ceiling.

        Role chaining caps the session at one hour, and the ExternalId is what
        stops another customer who learned the role ARN from steering this
        deputy at it.

        Ref: https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html
        """
        credential = tenant_signing.aws_credential
        assert credential is not None
        credentials = await tenant_role_credentials(_KEY_ID, credential)
        await credentials.get_frozen_credentials()

        assert fake_sts.calls == [
            {
                "RoleArn": _ROLE_ARN,
                "RoleSessionName": f"stdapi-ai-tenant-{_KEY_ID}",
                "ExternalId": _EXTERNAL_ID,
                "DurationSeconds": 3600,
            }
        ]

    async def test_the_session_is_cached_per_tenant(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """A second request reuses the session: one STS call per tenant per hour."""
        credential = tenant_signing.aws_credential
        assert credential is not None
        first = await tenant_role_credentials(_KEY_ID, credential)
        second = await tenant_role_credentials(_KEY_ID, credential)

        assert first is second
        assert len(fake_sts.calls) == 1

    async def test_concurrent_first_use_opens_one_session(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """A burst of first requests shares one AssumeRole call, not one each.

        Anything else would multiply the STS traffic by the concurrency of
        the tenant's first moment, throttling AWS STS for every tenant.
        """
        credential = tenant_signing.aws_credential
        assert credential is not None
        results = await gather(
            *(tenant_role_credentials(_KEY_ID, credential) for _ in range(5))
        )

        assert len({id(result) for result in results}) == 1
        assert len(fake_sts.calls) == 1

    async def test_two_tenants_open_distinct_sessions(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """Two keys sharing a role ARN get their own session and session name.

        Sharing one would misattribute the second tenant's calls to the first
        in AWS CloudTrail, whose principal carries the RoleSessionName.
        """
        credential = tenant_signing.aws_credential
        assert credential is not None
        other_key_id = "T" + "1" * 15
        first = await tenant_role_credentials(_KEY_ID, credential)
        second = await tenant_role_credentials(other_key_id, credential)

        assert first is not second
        assert [call["RoleSessionName"] for call in fake_sts.calls] == [
            f"stdapi-ai-tenant-{_KEY_ID}",
            f"stdapi-ai-tenant-{other_key_id}",
        ]

    async def test_a_near_expiry_session_is_reopened(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """A session close to its expiry re-assumes instead of dying mid-request."""
        credential = tenant_signing.aws_credential
        assert credential is not None
        fake_sts.expires_in = 60.0
        first = await tenant_role_credentials(_KEY_ID, credential)
        second = await tenant_role_credentials(_KEY_ID, credential)

        assert second is first
        assert len(fake_sts.calls) == 2

    async def test_the_cache_is_bounded_lru(
        self,
        tenant_signing: Tenant,
        fake_sts: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The least recently used tenant's session goes when the cache is full.

        An unbounded cache grows with the tenant population; evicting the
        wrong entry would turn every request into an AssumeRole call.
        """
        monkeypatch.setattr(aws, "_TENANT_ROLE_CACHE_MAX", 2)
        credential = tenant_signing.aws_credential
        assert credential is not None
        for key_id in ("A" * 16, "B" * 16, "C" * 16):
            await tenant_role_credentials(key_id, credential)
        assert len(aws._TENANT_ROLE_CACHE) == 2  # noqa: SLF001

        # The newest entry survived; the oldest was evicted and re-assumes.
        await tenant_role_credentials("C" * 16, credential)
        assert len(fake_sts.calls) == 3
        await tenant_role_credentials("A" * 16, credential)
        assert len(fake_sts.calls) == 4

    async def test_a_rotated_credential_opens_a_fresh_session(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """Changing the registered role must not reuse the old role's session."""
        credential = tenant_signing.aws_credential
        assert credential is not None
        await tenant_role_credentials(_KEY_ID, credential)
        rotated = TenantAwsCredential(
            role_arn=_ROLE_ARN, external_id="rotated-external-id"
        )
        await tenant_role_credentials(_KEY_ID, rotated)

        assert len(fake_sts.calls) == 2
        assert fake_sts.calls[1]["ExternalId"] == "rotated-external-id"

    async def test_a_refused_assume_role_is_an_honest_403(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace
    ) -> None:
        """A wrong ExternalId (or revoked trust) answers a fixed, leak-free 403.

        The AWS failure names the gateway's assumed-role principal; none of it
        may reach the client. The failed entry is dropped so the next request
        retries with a fresh AssumeRole rather than a poisoned cache entry.
        """
        fake_sts.failure = ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": (
                        f"User: arn:aws:sts::{_GATEWAY_ACCOUNT}:assumed-role/"
                        "stdapi-gateway/task is not authorized to perform: "
                        f"sts:AssumeRole on resource: {_ROLE_ARN}"
                    ),
                }
            },
            "AssumeRole",
        )
        credential = tenant_signing.aws_credential
        assert credential is not None
        with pytest.raises(ApiError) as excinfo:
            await tenant_role_credentials(_KEY_ID, credential)

        assert excinfo.value.status == 403
        assert str(excinfo.value) == TENANT_CREDENTIAL_FAILURE_MESSAGE
        assert not _INTERNAL_DETAIL_RE.search(str(excinfo.value))
        assert excinfo.value.__cause__ is None

        # The failure was not cached: the next attempt calls AWS STS again.
        fake_sts.failure = None
        await tenant_role_credentials(_KEY_ID, credential)
        assert len(fake_sts.calls) == 2

    @pytest.mark.parametrize(
        "failure",
        [
            ClientError({"Error": {"Code": "Throttling"}}, "AssumeRole"),
            ClientError({"Error": {"Code": "ServiceUnavailable"}}, "AssumeRole"),
            EndpointConnectionError(endpoint_url="https://sts.us-east-1.amazonaws.com"),
        ],
        ids=["throttled", "unavailable", "unreachable"],
    )
    async def test_a_transient_assume_role_failure_is_a_retryable_503(
        self, tenant_signing: Tenant, fake_sts: SimpleNamespace, failure: Exception
    ) -> None:
        """AWS STS being throttled or unreachable is this deployment's outage.

        Only the codes naming the registration itself are the tenant's fault. A
        throttled or unreachable AWS STS clears on its own, and answering it
        with the tenant's 403 both sends the tenant auditing a trust policy
        that is fine and stops its SDK retrying: OpenAI and Anthropic clients
        retry 408, 409, 429 and 5xx, never a 403.

        Ref: https://docs.aws.amazon.com/STS/latest/APIReference/CommonErrors.html
             stdapi/aws.py:tenant_role_credentials
        """
        fake_sts.failure = failure
        credential = tenant_signing.aws_credential
        assert credential is not None
        with pytest.raises(ApiError) as excinfo:
            await tenant_role_credentials(_KEY_ID, credential)

        assert excinfo.value.status == 503
        assert str(excinfo.value) == TENANT_SESSION_UNAVAILABLE_MESSAGE
        assert not _INTERNAL_DETAIL_RE.search(str(excinfo.value))
        assert excinfo.value.__cause__ is None

    async def test_the_tenant_credential_outranks_the_user_role(
        self,
        tenant_signing: Tenant,
        fake_sts: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With both features on, the tenant's own account signs the call.

        The user role only attributes spend within the gateway's account; the
        tenant credential moves it to another one, which must win.
        """
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_user_role_arn",
            f"arn:aws:iam::{_GATEWAY_ACCOUNT}:role/end-user",
        )
        log: dict[str, Any] = {}
        token = REQUEST_LOG.set(log)  # type: ignore[arg-type]
        try:
            credentials = await request_signing_credentials()
        finally:
            REQUEST_LOG.reset(token)

        assert credentials is not None
        assert fake_sts.calls[0]["RoleArn"] == _ROLE_ARN
        assert log["aws_tenant_key_id"] == _KEY_ID

    async def test_without_a_credential_nothing_is_signed_specially(self) -> None:
        """No tenant credential and no user role keeps the server's identity."""
        assert await request_signing_credentials() is None

    def test_the_predicate_requires_the_request_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``signed_as_tenant`` is scoped to signed operations of this request."""
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        token = REQUEST_LOG.set(make_event_log(aws_tenant_key_id=_KEY_ID))
        try:
            assert signed_as_tenant("Converse")
            assert not signed_as_tenant("CreateModelInvocationJob")
        finally:
            REQUEST_LOG.reset(token)
        assert not signed_as_tenant("Converse")


class TestSignedRequest:
    """The pooled client signs the invocation with the tenant's session.

    The tenant twin of ``tests/test_aws_identity.py::TestSignedRequest``:
    botocore's ``request_credentials`` contract and the guard of the
    ``before-parameter-build`` hook are what actually move the signature --
    and the bill -- onto the tenant's account, so the headers a pooled client
    would really send are asserted, with no user role configured at all.

    Ref: stdapi/aws.py:_sign_model_invocation
    """

    class _AbortError(Exception):
        """Stops the request before it leaves the process."""

    async def _signed_headers(self, operation: str = "Converse") -> dict[str, Any]:
        """Return the headers of the request a Bedrock runtime call would send.

        Args:
            operation: The Bedrock runtime operation to prepare.

        Returns:
            The raw headers botocore signed, captured before any send.
        """
        headers: dict[str, Any] = {}

        def _capture(request: AWSPreparedRequest, **_kwargs: object) -> None:
            headers.update(request.headers.items())
            raise self._AbortError

        async with AWS_SESSION.create_client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=_SERVER_KEY,
            aws_secret_access_key="server-secret",  # noqa: S106
            config=AioConfig(parameter_validation=False, retries={"max_attempts": 1}),
        ) as client:
            client.meta.events.register("before-send", _capture)
            call = {
                "Converse": lambda: client.converse(
                    modelId="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": [{"text": "hi"}]}],
                ),
                "ConverseStream": lambda: client.converse_stream(
                    modelId="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": [{"text": "hi"}]}],
                ),
                "ApplyGuardrail": lambda: client.apply_guardrail(
                    guardrailIdentifier="gr",
                    guardrailVersion="1",
                    source="INPUT",
                    content=[{"text": {"text": "hi"}}],
                ),
            }[operation]
            with pytest.raises(self._AbortError):
                await call()  # type: ignore[no-untyped-call]
        return headers

    @pytest.mark.parametrize("operation", ["Converse", "ConverseStream"])
    async def test_an_invocation_is_signed_with_the_tenant_session(
        self,
        tenant_signing: Tenant,
        fake_sts: SimpleNamespace,
        request_log: dict[str, Any],
        operation: str,
    ) -> None:
        """The signature carries the tenant session's key, and marks the request.

        No user role is configured, so the hook must fire on the tenant
        credential alone -- or every invocation silently signs, and bills,
        as the deployment while the usage log claims the tenant paid.
        """
        headers = await self._signed_headers(operation)

        assert b"Credential=AKIDTEST/" in headers["Authorization"]
        assert _SERVER_KEY.encode() not in headers["Authorization"]
        assert headers["X-Amz-Security-Token"] == b"token"
        assert request_log["aws_tenant_key_id"] == _KEY_ID

    async def test_other_operations_keep_the_server_identity(
        self,
        tenant_signing: Tenant,
        fake_sts: SimpleNamespace,
        request_log: dict[str, Any],
    ) -> None:
        """A non-invocation operation signs as the server and opens no session.

        Anything else would require every tenant's role to carry the
        deployment's own auxiliary permissions too.
        """
        headers = await self._signed_headers("ApplyGuardrail")

        assert f"Credential={_SERVER_KEY}/".encode() in headers["Authorization"]
        assert fake_sts.calls == []
        assert "aws_tenant_key_id" not in request_log


class TestErrorClassification:
    """What a tenant reads when its own account refuses the call.

    Ref: stdapi/api_errors.py:_tenant_signed_denial
    """

    @pytest.fixture
    def tenant_marked_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[dict[str, Any]]:
        """Install a request log marked as tenant-signed.

        Yields:
            The request log.
        """
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        log: dict[str, Any] = {"aws_tenant_key_id": _KEY_ID}
        token = REQUEST_LOG.set(log)  # type: ignore[arg-type]
        try:
            yield log
        finally:
            REQUEST_LOG.reset(token)

    @staticmethod
    def _client_error(code: str) -> ClientError:
        """Build a ClientError whose message is full of gateway details.

        Args:
            code: The AWS error code.

        Returns:
            The error, as botocore would raise it for a Converse call.
        """
        return ClientError(
            {
                "Error": {
                    "Code": code,
                    "Message": (
                        f"User: arn:aws:sts::{_GATEWAY_ACCOUNT}:assumed-role/"
                        "stdapi-gateway/task is not authorized to perform: "
                        "bedrock:InvokeModel"
                    ),
                }
            },
            "Converse",
        )

    def test_a_denied_invocation_is_the_tenants_own_403(
        self, tenant_marked_request: dict[str, Any]
    ) -> None:
        """AccessDenied under tenant signing is a fixed 403, not a 503 outage.

        The body names none of the gateway's account details: no ARN, no
        account ID, no AWS error code.
        """
        error = denied_feature_unavailable(self._client_error("AccessDeniedException"))

        assert isinstance(error, ApiError)
        assert error.status == 403
        assert str(error) == TENANT_ACCESS_DENIED_MESSAGE
        assert not _INTERNAL_DETAIL_RE.search(str(error))

    def test_a_dead_session_is_a_403_and_drops_the_cache(
        self, tenant_marked_request: dict[str, Any]
    ) -> None:
        """An expired session answers 403 and evicts the tenant's sessions.

        Never a 401: the caller's API key is fine, the gateway's session died.
        Evicted so the next request opens a fresh session instead of re-signing
        with the dead one for up to an hour.
        """
        aws._TENANT_ROLE_CACHE["stale"] = SimpleNamespace(key_id=_KEY_ID)  # type: ignore[assignment] # noqa: SLF001
        try:
            error = denied_feature_unavailable(
                self._client_error("ExpiredTokenException")
            )

            assert isinstance(error, ApiError)
            assert error.status == 403
            assert str(error) == TENANT_CREDENTIAL_FAILURE_MESSAGE
            assert not _INTERNAL_DETAIL_RE.search(str(error))
            assert "stale" not in aws._TENANT_ROLE_CACHE  # noqa: SLF001
        finally:
            aws.clear_tenant_role_cache()

    def test_a_server_denial_stays_a_feature_unavailable(self) -> None:
        """Without the tenant marker, the operator-misconfiguration path holds."""
        error = denied_feature_unavailable(self._client_error("AccessDeniedException"))

        assert error is not None
        assert error.status == 503


class TestTheRefusalReachesTheTenant:
    """The two tenant 403 texts are sent as written, on every terminal path.

    A 403 body is otherwise replaced by the bare word "Forbidden" before it
    leaves the process -- right for a refusal of this deployment's own identity,
    and wrong for these two, which are written for the tenant, name only the
    tenant's own resources, and are what the documentation tells an operator to
    tell the two apart by.

    Ref: https://stdapi.ai/operations_authentication_security/#tenant-aws-credentials
         https://stdapi.ai/operations_troubleshooting/
         stdapi/utils.py:hide_security_details
    """

    @staticmethod
    def _connection(path: str) -> HTTPConnection:
        """Return a bare connection for *path*, enough to pick the envelope."""
        return HTTPConnection(
            {"type": "http", "method": "POST", "path": path, "headers": []}
        )

    @pytest.mark.parametrize(
        "message", [TENANT_CREDENTIAL_FAILURE_MESSAGE, TENANT_ACCESS_DENIED_MESSAGE]
    )
    async def test_the_rest_body_carries_the_message(self, message: str) -> None:
        """The JSON error envelope of a REST route carries the text itself."""
        from stdapi.main import handle_api_error  # noqa: PLC0415

        token = REQUEST_LOG.set(make_event_log())
        try:
            response = await handle_api_error(
                self._connection("/v1/chat/completions"),  # type: ignore[arg-type]
                TenantCredentialError(message),
            )
        finally:
            REQUEST_LOG.reset(token)

        assert response.status_code == 403
        assert message in bytes(response.body).decode()

    @pytest.mark.parametrize(
        "message", [TENANT_CREDENTIAL_FAILURE_MESSAGE, TENANT_ACCESS_DENIED_MESSAGE]
    )
    def test_the_streamed_error_event_carries_the_message(self, message: str) -> None:
        """A refusal ending a stream says the same thing the REST body does."""
        from stdapi.monitoring import REQUEST, _api_error_sse_event  # noqa: PLC0415

        log_token = REQUEST_LOG.set(make_event_log())
        request_token = REQUEST.set(self._connection("/v1/chat/completions"))
        try:
            event = _api_error_sse_event(TenantCredentialError(message))
        finally:
            REQUEST.reset(request_token)
            REQUEST_LOG.reset(log_token)

        assert message in str(event.data)

    def test_any_other_403_is_still_flattened(self) -> None:
        """Every 403 this exemption does not name keeps saying only "Forbidden".

        The exemption is the class, not the status: a refusal carrying an AWS
        message must not start leaking one because of it.
        """
        denied = "Denied on arn:aws:iam::123456789012:role/stdapi-gateway"
        assert hide_security_details(403, denied) == "Forbidden"
        assert hide_security_details(401, denied) == "Unauthorized"


class TestRegionRouterIsolation:
    """A tenant's throttle must not write the shared per-region backoff.

    Bedrock quota is per account per region: the tenant's throttle says
    nothing about the operator's account, and writing it would let one
    throttled tenant put a region on backoff for every other caller.

    Ref: stdapi/models/__init__.py:route_and_execute
         https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html
    """

    @staticmethod
    def _throttle_then_succeed() -> Callable[[str], Awaitable[str]]:
        """Return an fn throttled in the first region, serving in the second."""

        async def fn(region: str) -> str:
            if region == "us-east-1":
                raise ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "throttled"}},
                    "Converse",
                )
            return f"served-in-{region}"

        return fn

    async def test_a_tenant_throttle_fails_over_without_marking(
        self, tenant_signing: Tenant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tenant request fails over, and the shared state stays clean."""
        router = RegionRouter()
        monkeypatch.setattr(stdapi.models, "REGION_ROUTER", router)

        result = await route_and_execute(
            "model-under-test",
            ["us-east-1", "us-west-2"],
            self._throttle_then_succeed(),
        )

        assert result == "served-in-us-west-2"
        state = router._index.get("model-under-test", "us-east-1")  # noqa: SLF001
        assert state.consecutive_quota_errors == 0
        assert state.quota_blocked_until == 0.0

    async def test_an_operator_throttle_still_marks_the_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a tenant credential the shared bookkeeping is unchanged."""
        router = RegionRouter()
        monkeypatch.setattr(stdapi.models, "REGION_ROUTER", router)

        result = await route_and_execute(
            "model-under-test",
            ["us-east-1", "us-west-2"],
            self._throttle_then_succeed(),
        )

        assert result == "served-in-us-west-2"
        state = router._index.get("model-under-test", "us-east-1")  # noqa: SLF001
        assert state.consecutive_quota_errors == 1
        assert state.quota_blocked_until > 0.0


class TestBillableServicePins:
    """Services the tenant's credential cannot pay for are pinned or refused.

    Ref: stdapi/models/__init__.py:_pin_tenant_billable_service
    """

    @staticmethod
    def _details(
        model_id: str, service: str, output_modality: str = "TEXT"
    ) -> ModelDetails:
        """Build minimal model details on *service*.

        Args:
            model_id: The model identifier.
            service: The hosting service label.
            output_modality: The advertised output modality.

        Returns:
            The details.
        """
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="OpenAI",
            service=service,
            input_modalities=["TEXT"],
            output_modalities=[output_modality],
            regions=["us-east-1"],
        )

    def test_a_dual_homed_mantle_model_pins_to_its_runtime_twin(
        self, tenant_signing: Tenant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Mantle-default model (the GPT-5.6 family) serves via bedrock-runtime.

        There the tenant's credential signs and pays; Mantle would ride the
        gateway's session and bill the operator.
        """
        mantle = self._details("openai.gpt-5.6", MANTLE_SERVICE)
        twin = self._details("openai.gpt-5.6-runtime", "AWS Bedrock Runtime")
        monkeypatch.setitem(
            stdapi.models._MANTLE_RUNTIME_TWINS,  # noqa: SLF001
            mantle.id,
            twin.id,
        )
        monkeypatch.setitem(stdapi.models._ALL_MODELS, twin.id, twin)  # noqa: SLF001

        pinned, pinned_id = stdapi.models._pin_tenant_billable_service(  # noqa: SLF001
            mantle, mantle.id
        )

        assert pinned is twin
        assert pinned_id == twin.id

    def test_a_mantle_only_model_is_refused(self, tenant_signing: Tenant) -> None:
        """A model with no runtime home is refused, naming the model."""
        mantle = self._details("openai.mantle-only", MANTLE_SERVICE)

        with pytest.raises(ApiError, match=re.escape("openai.mantle-only")) as excinfo:
            stdapi.models._pin_tenant_billable_service(mantle, mantle.id)  # noqa: SLF001

        assert "not available" in str(excinfo.value)

    def test_a_marketplace_endpoint_is_refused(self, tenant_signing: Tenant) -> None:
        """A Marketplace endpoint is the operator's provisioned resource."""
        endpoint = self._details("listed.model", MARKETPLACE_SERVICE)

        with pytest.raises(ApiError, match="Marketplace"):
            stdapi.models._pin_tenant_billable_service(endpoint, endpoint.id)  # noqa: SLF001

    def test_without_a_credential_nothing_is_pinned(self) -> None:
        """A plain request serves Mantle models exactly as before."""
        mantle = self._details("openai.gpt-5.6", MANTLE_SERVICE)

        assert stdapi.models._pin_tenant_billable_service(  # noqa: SLF001
            mantle, mantle.id
        ) == (mantle, mantle.id)


class TestServiceRefusals:
    """Operations a tenant credential cannot cover refuse loudly, upfront.

    Ref: stdapi/batches.py:create_batch
         stdapi/aws.py:verify_bidi_user_role_policy
    """

    async def test_a_batch_is_refused_before_any_job_starts(
        self, tenant_signing: Tenant
    ) -> None:
        """A batch job outlives the 1 h session; starting one would misbill.

        Refused with a clear reason rather than started on the operator's
        account, or started only to fail hours later. The match is the
        tenant-specific wording and the 400: the unrelated batches-disabled
        refusal shares the "Batch API is not available" prefix and is a 503,
        and must not be able to satisfy this test.
        """
        from stdapi.batches import create_batch  # noqa: PLC0415

        with pytest.raises(
            ApiError, match="carry an AWS credential of their own"
        ) as excinfo:
            await create_batch(
                surface="openai",
                endpoint="/v1/chat/completions",
                completion_window="24h",
                prepared=[],
            )

        assert excinfo.value.status == 400

    def test_a_realtime_model_session_is_refused(self, tenant_signing: Tenant) -> None:
        """A bidirectional stream signs once, as the server: refuse it."""
        with pytest.raises(ApiError, match="real-time") as excinfo:
            verify_bidi_user_role_policy("bedrock-runtime")
        assert str(excinfo.value) == TENANT_REALTIME_MESSAGE

    def test_non_model_streams_are_unaffected(self, tenant_signing: Tenant) -> None:
        """Only model invocation streams are the tenant's to pay for."""
        verify_bidi_user_role_policy("transcribe-streaming")

    def test_without_a_credential_realtime_is_untouched(self) -> None:
        """A plain deployment keeps serving realtime sessions."""
        verify_bidi_user_role_policy("bedrock-runtime")

    async def test_asynchronous_generation_is_refused(
        self, tenant_signing: Tenant
    ) -> None:
        """StartAsyncInvoke writes to the gateway's bucket on its account."""
        from stdapi.models.video.amazon_nova_reel import VideoModel  # noqa: PLC0415

        with pytest.raises(ApiError, match="asynchronously"):
            await VideoModel("amazon.nova-reel-v1:1").invoke_async({})

    def test_the_mantle_service_header_is_refused(
        self, tenant_signing: Tenant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-request Mantle header would bill the operator: refuse it."""
        from stdapi.models.chat import serves_via_mantle  # noqa: PLC0415
        from stdapi.monitoring import REQUEST  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        monkeypatch.setitem(stdapi.models.MANTLE_MODELS, "openai.gpt-5.6", object())
        request = SimpleNamespace(headers={"x-stdapi-service": "bedrock-mantle"})
        token = REQUEST.set(request)  # type: ignore[arg-type]
        try:
            with pytest.raises(ApiError, match="x-stdapi-service"):
                serves_via_mantle("openai.gpt-5.6")
        finally:
            REQUEST.reset(token)


class TestUsageBilling:
    """Tenant-billed usage is marked and never priced as the operator's spend.

    Ref: stdapi/usage.py:record_bedrock_usage
    """

    @pytest.fixture
    def usage_scope(self) -> Iterator[None]:
        """Install a fresh per-request usage map.

        Yields:
            None.
        """
        token = stdapi.usage.init_usage()
        try:
            yield
        finally:
            stdapi.usage.USAGE.reset(token)

    def test_a_tenant_signed_invocation_is_marked_and_unpriced(
        self, tenant_signing: Tenant, usage_scope: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record says who was billed, and no cost is claimed for it.

        The mark follows the request-log marker written at signing time, not
        the credential's mere presence: it says the invocation really ran on
        the tenant's account.
        """
        monkeypatch.setattr(stdapi.usage, "price_catalog_ready", lambda: True)
        monkeypatch.setattr(
            stdapi.usage,
            "resolve_price",
            lambda *_args, **_kwargs: SimpleNamespace(
                currency="USD", amount=Decimal("0.001")
            ),
        )
        token = REQUEST_LOG.set(make_event_log(aws_tenant_key_id=_KEY_ID))
        try:
            stdapi.usage.record_bedrock_usage(
                "amazon.nova-lite-v1:0", region="us-east-1", input_tokens=1000
            )
        finally:
            REQUEST_LOG.reset(token)

        warnings = stdapi.usage.compute_costs()
        entries = stdapi.usage.usage_log_entries()

        assert warnings == []
        assert len(entries) == 1
        assert entries[0]["billed_to"] == "tenant"
        assert "cost" not in entries[0]
        assert stdapi.usage.total_costs_by_currency(entries) == {}

    def test_a_forced_operator_record_is_not_marked(
        self, tenant_signing: Tenant, usage_scope: None
    ) -> None:
        """A batch's results stay the operator's spend whoever reads them."""
        stdapi.usage.record_bedrock_usage(
            "amazon.nova-lite-v1:0",
            tier="batch",
            region="us-east-1",
            input_tokens=1000,
            billed_externally=False,
        )

        assert all(
            "billed_to" not in entry for entry in stdapi.usage.usage_log_entries()
        )

    def test_a_server_signed_bedrock_record_stays_unmarked(
        self, tenant_signing: Tenant, usage_scope: None
    ) -> None:
        """A Bedrock call the tenant session never signed is the operator's spend.

        Without the request-log marker the credential's presence proves
        nothing: marking such a record tenant-billed would drop real operator
        spend from the deployment's cost totals.
        """
        token = REQUEST_LOG.set(make_event_log())
        try:
            stdapi.usage.record_bedrock_usage(
                "cohere.rerank-v3-5:0", region="us-east-1", search_units=3
            )
        finally:
            REQUEST_LOG.reset(token)

        assert all(
            "billed_to" not in entry for entry in stdapi.usage.usage_log_entries()
        )

    def test_gateway_signed_services_stay_unmarked(
        self, tenant_signing: Tenant, usage_scope: None
    ) -> None:
        """Auxiliary spend in the same request stays on the operator's bill."""
        stdapi.usage.record_polly_usage(100, "neural", region="us-east-1")
        stdapi.usage.record_guardrail_policy_usage(
            {"contentPolicyUnits": 3}, region="us-east-1"
        )

        entries = stdapi.usage.usage_log_entries()
        assert entries
        assert all("billed_to" not in entry for entry in entries)

    def test_without_a_credential_nothing_is_marked(self, usage_scope: None) -> None:
        """A plain request's records are exactly as before."""
        stdapi.usage.record_bedrock_usage(
            "amazon.nova-lite-v1:0", region="us-east-1", input_tokens=1000
        )

        assert all(
            "billed_to" not in entry for entry in stdapi.usage.usage_log_entries()
        )


class TestCredentialHelper:
    """The request-scope credential resolution helper.

    Ref: stdapi/monitoring.py:tenant_aws_credential
    """

    def test_disabled_feature_hides_the_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the setting off, a credential-bearing tenant signs nothing."""
        token = TENANT.set(
            _tenant(TenantAwsCredential(role_arn=_ROLE_ARN, external_id="x"))
        )
        try:
            monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", False)
            assert tenant_aws_credential() is None
            monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
            assert tenant_aws_credential() is not None
        finally:
            TENANT.reset(token)

    def test_a_plain_tenant_carries_no_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tenant without a registered role never triggers tenant signing."""
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)
        token = TENANT.set(_tenant())
        try:
            assert tenant_aws_credential() is None
        finally:
            TENANT.reset(token)
