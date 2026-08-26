r"""Per-end-user role sessions for Amazon Bedrock model invocations.

The gateway optionally signs a model invocation with credentials of a role
session opened for the end user, so AWS reports the spend under that user
instead of under the server's own identity. Three contracts are load-bearing
and all three are asserted here rather than assumed:

- AWS STS accepts a role session name of 2-64 characters over ``[\w+=,.@-]``
  and a session tag value of at most 256 characters over
  ``[\p{L}\p{Z}\p{N}_.:/=+\-@]``, so an end user identifier must be mapped
  into each of them separately, and two identifiers must never map to one
  session: session tags are enforceable through ``aws:PrincipalTag``, which
  makes a collision an authorization defect and not only a billing one.
- botocore signs a request with ``context["signing"]["request_credentials"]``
  when one is present, which is what lets a single pooled client sign each
  request as a different identity.
- A failure to open the session must reach the client as a server-side error
  carrying nothing from AWS, and must never be mistaken for a per-region
  Bedrock failure.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-iam-principal-tracking.html
     https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/iam-principal-cost-allocation.html
     https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
     stdapi/aws.py:user_role_credentials
"""

import re
from asyncio import gather, sleep
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Self

import pytest
from aiobotocore.config import AioConfig
from botocore.loaders import create_loader

import stdapi.aws
from stdapi.api_errors import ApiError
from stdapi.aws import (
    USER_ROLE_OPERATIONS,
    user_role_credentials,
    user_role_session_identity,
)
from stdapi.config import AWS_SESSION, SETTINGS, _Settings
from stdapi.monitoring import PRINCIPAL, Principal
from tests._helpers import make_client_error, make_event_log

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from botocore.awsrequest import AWSPreparedRequest

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: A syntactically valid role ARN; no call ever reaches AWS in this module.
_ROLE_ARN = "arn:aws:iam::123456789012:role/stdapi-ai-end-user"

#: Access key the pooled client itself is built on (the server's own identity).
_SERVER_KEY = "AKIASERVERROLEXXXXXX"

#: Access key the stubbed role session returns (the end user's identity).
_USER_KEY = "AKIAENDUSERXXXXXXXXX"


#: AWS STS wire model, the authority for the AssumeRole limits asserted here.
_STS_SHAPES: dict[str, Any] = create_loader().load_service_model("sts", "service-2")[
    "shapes"
]

#: AWS STS RoleSessionName limits, read from the wire model.
_SESSION_NAME_SHAPE = _STS_SHAPES["roleSessionNameType"]

#: AWS STS session tag value limits, read from the wire model.
_TAG_VALUE_SHAPE = _STS_SHAPES["tagValueType"]

#: RoleSessionName charset, directly usable: its wire pattern is ASCII already.
_SESSION_NAME_RE = re.compile(f"^(?:{_SESSION_NAME_SHAPE['pattern']})$")

#: Tag value charset: the wire model's Unicode classes as Python equivalents.
_TAG_VALUE_RE = re.compile(r"^[\w .:/=+\-@]*$")


@pytest.fixture(autouse=True)
def _isolated_cache() -> Iterator[None]:
    """Run each test on an empty role-session cache."""
    stdapi.aws.clear_user_role_cache()
    yield
    stdapi.aws.clear_user_role_cache()


@pytest.fixture
def user_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable per-end-user role sessions with the default tag key."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", _ROLE_ARN)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_tag_key", "user")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_session_duration", 3600)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", False)


def _stub_assume_role(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expires_in: float = 3600,
    error: Exception | None = None,
) -> list[tuple[str, str | None]]:
    """Replace the AWS STS call with a stub, returning its recorded arguments.

    The stub suspends before answering, as the real call does: without a
    suspension point a burst of callers runs strictly one after another, and
    nothing here would exercise what happens while a session is being opened.
    """
    calls: list[tuple[str, str | None]] = []

    async def _assume_role(session_name: str, tag_value: str | None) -> dict[str, Any]:
        calls.append((session_name, tag_value))
        await sleep(0)
        if error is not None:
            raise error
        return {
            "access_key": _USER_KEY,
            "secret_key": "user-secret",
            "token": "user-token",
            "expiry_time": (
                datetime.now(UTC) + timedelta(seconds=expires_in)
            ).isoformat(),
        }

    monkeypatch.setattr(stdapi.aws, "_assume_user_role", _assume_role)
    return calls


class TestSessionIdentity:
    """The end user identifier is mapped into what AWS STS accepts, injectively.

    Ref: https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
         stdapi/aws.py:user_role_session_identity
    """

    #: Identifiers AWS STS rejects verbatim, plus two that collide when stripped.
    IDENTITIES = (
        "alice",
        "alicé",
        "alice smith",
        "user:alice",
        "al*ice",
        "al?ice",
        "*",
        "",
        "a" * 300,
        "user@example.com",
        "arn:aws:iam::123456789012:user/alice",
        "8f14e45f-ceea-467a-9c4f-0b0f0ba0d2e1",
    )

    def test_wire_model_still_states_the_limits_assumed_here(self) -> None:
        """The STS limits this mapping targets are the ones the SDK declares."""
        assert _SESSION_NAME_SHAPE["min"] == 2
        assert _SESSION_NAME_SHAPE["max"] == 64
        assert _SESSION_NAME_SHAPE["pattern"] == r"[\w+=,.@-]*"
        assert _TAG_VALUE_SHAPE["max"] == 256
        assert _TAG_VALUE_SHAPE["pattern"] == r"[\p{L}\p{Z}\p{N}_.:/=+\-@]*"

    @pytest.mark.parametrize("identity", IDENTITIES)
    def test_session_name_is_accepted_by_sts(self, identity: str) -> None:
        """Every identifier maps to a session name inside the STS charset and length."""
        session_name, _ = user_role_session_identity(identity)
        assert 2 <= len(session_name) <= 64
        assert _SESSION_NAME_RE.fullmatch(session_name)

    @pytest.mark.parametrize("identity", IDENTITIES)
    def test_tag_value_is_accepted_by_sts_and_never_empty(self, identity: str) -> None:
        """Every identifier maps to a non-empty tag value inside the STS charset.

        STS accepts an empty tag value, which would silently pool every
        unmappable identifier into a single billing bucket.
        """
        _, tag_value = user_role_session_identity(identity)
        assert 0 < len(tag_value) <= 256
        assert _TAG_VALUE_RE.fullmatch(tag_value)

    def test_distinct_identities_never_share_a_session(self) -> None:
        """Identifiers differing only in rejected characters stay distinct.

        Session tags are enforceable as ``aws:PrincipalTag``, so two end users
        sharing a session is an authorization defect, not only a billing one.
        """
        names = {user_role_session_identity(i)[0] for i in self.IDENTITIES}
        values = {user_role_session_identity(i)[1] for i in self.IDENTITIES}
        assert len(names) == len(self.IDENTITIES)
        assert len(values) == len(self.IDENTITIES)

    def test_mapping_is_stable(self) -> None:
        """The same identifier always maps to the same session, across processes.

        AWS reports one Cost and Usage Report row per session name, so an
        unstable mapping would scatter one user's spend across many rows.
        """
        assert user_role_session_identity("alice") == user_role_session_identity(
            "alice"
        )
        assert user_role_session_identity("alice")[0].startswith("alice-")

    def test_readable_identity_survives_into_both_forms(self) -> None:
        """A readable identifier stays recognisable in the session name and tag."""
        session_name, tag_value = user_role_session_identity("alice@example.com")
        assert session_name.startswith("alice@example.com-")
        assert tag_value.startswith("alice@example.com-")


class TestCredentialCache:
    """One role session per end user, refreshed before expiry and bounded in size.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-iam-principal-tracking.html
         stdapi/aws.py:user_role_credentials
    """

    async def test_concurrent_first_calls_open_one_session(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A burst of first requests for one end user issues a single AssumeRole.

        AWS documents caching as a requirement of this design: the STS request
        quota is shared account-wide and per Region.

        The eight callers must genuinely overlap, which ``peak`` asserts: with
        a stub that never suspends they would run one after another, and plain
        caching would satisfy the single-call assertion.
        """
        calls = _stub_assume_role(monkeypatch)
        pending = peak = 0

        async def _call() -> Any:  # noqa: ANN401
            nonlocal pending, peak
            pending += 1
            peak = max(peak, pending)
            try:
                return await user_role_credentials("alice")
            finally:
                pending -= 1

        results = await gather(*(_call() for _ in range(8)))
        assert peak == 8
        assert len(calls) == 1
        for credentials in results:
            assert (await credentials.get_frozen_credentials()).access_key == _USER_KEY

    async def test_hostile_identifier_is_not_what_the_cache_keeps(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache entry of an end user is bounded, whatever the request declares.

        An end user identifier is client-chosen and has no length limit, so a
        cache keyed by it would be bounded in entries but not in memory.
        """
        calls = _stub_assume_role(monkeypatch)
        identity = "a" * 1_000_000
        await user_role_credentials(identity)
        assert all(len(key) <= 64 for key in stdapi.aws._USER_ROLE_CACHE)  # noqa: SLF001
        await user_role_credentials(identity)
        assert len(calls) == 1
        await user_role_credentials(identity + "b")
        assert len(calls) == 2

    async def test_cached_session_is_reused(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session still far from expiry is reused instead of reopened."""
        calls = _stub_assume_role(monkeypatch)
        for _ in range(5):
            await user_role_credentials("alice")
        assert len(calls) == 1

    async def test_each_identity_gets_its_own_session(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two end users never share a role session."""
        calls = _stub_assume_role(monkeypatch)
        await user_role_credentials("alice")
        await user_role_credentials("bob")
        assert len(calls) == 2
        assert calls[0] != calls[1]

    async def test_short_session_is_not_refreshed_on_every_call(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refresh window scales with the session, instead of botocore's fixed 15 minutes.

        botocore refreshes a fixed 900 seconds before expiry, which at the
        900-second minimum means one AssumeRole per request.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_session_duration", 900)
        calls = _stub_assume_role(monkeypatch, expires_in=300)
        await user_role_credentials("alice")
        await user_role_credentials("alice")
        assert len(calls) == 1

    async def test_session_is_refreshed_before_it_expires(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session close to expiry is reopened before it is used again."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_session_duration", 900)
        calls = _stub_assume_role(monkeypatch, expires_in=30)
        await user_role_credentials("alice")
        await user_role_credentials("alice")
        assert len(calls) == 2

    async def test_cache_is_bounded_and_evicts_least_recently_used(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A high-cardinality identity space cannot grow the cache without bound."""
        monkeypatch.setattr(stdapi.aws, "_USER_ROLE_CACHE_MAX", 4)
        calls = _stub_assume_role(monkeypatch)
        for identity in ("a", "b", "c", "d"):
            await user_role_credentials(identity)
        await user_role_credentials("a")  # refreshes "a" recency, evicting "b"
        await user_role_credentials("e")
        assert len(calls) == 5
        await user_role_credentials("a")
        assert len(calls) == 5
        await user_role_credentials("b")
        assert len(calls) == 6


class TestFailClosed:
    """An AWS STS failure fails the request, leaking nothing and failing over nowhere.

    Ref: stdapi/aws.py:user_role_credentials
         stdapi/aws_bedrock.py:AWS_ERROR_MAP
    """

    ERRORS = ("AccessDenied", "ThrottlingException", "ExpiredTokenException")

    @pytest.mark.parametrize("code", ERRORS)
    async def test_error_becomes_a_server_side_failure(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        """Every AWS STS error surfaces as a 5xx, never as a client credential error.

        ``AWS_ERROR_MAP`` renders ``ExpiredTokenException`` as 401 and
        ``AccessDeniedException`` as 403: told to the caller, both would blame
        the caller's credentials for a server-side misconfiguration.
        """
        _stub_assume_role(
            monkeypatch,
            error=make_client_error(
                code, "AssumeRole", message=f"User: arn:aws:sts::123456789012:{code}"
            ),
        )
        with pytest.raises(ApiError) as raised:
            await user_role_credentials("alice")
        assert raised.value.status >= 500

    @pytest.mark.parametrize("code", ERRORS)
    async def test_error_carries_no_aws_detail(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        """The message names no ARN, account, role or AWS error code."""
        _stub_assume_role(
            monkeypatch,
            error=make_client_error(
                code,
                "AssumeRole",
                message=f"User: arn:aws:sts::123456789012:assumed-role/x is not "
                f"authorized to perform: sts:AssumeRole on {_ROLE_ARN}",
            ),
        )
        with pytest.raises(ApiError) as raised:
            await user_role_credentials("alice")
        message = str(raised.value)
        for leak in ("arn:", "123456789012", "sts:", "AssumeRole", code, _ROLE_ARN):
            assert leak not in message

    @pytest.mark.parametrize("code", ERRORS)
    async def test_error_never_fails_the_request_over_to_another_region(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        """The Bedrock region router reads the failure as fatal, not as a bad Region.

        It classifies an ``ApiError`` by the AWS error chained to it, so an AWS
        STS throttle chained here would mark every Bedrock Region unhealthy and
        re-attempt the request -- and its role session -- in each of them.
        """
        from stdapi.models import _retryable_error_code  # noqa: PLC0415

        _stub_assume_role(monkeypatch, error=make_client_error(code, "AssumeRole"))
        with pytest.raises(ApiError) as raised:
            await user_role_credentials("alice")
        assert raised.value.__cause__ is None
        assert _retryable_error_code(raised.value) is None

    async def test_transient_failure_fails_the_request_and_not_the_end_user(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled AssumeRole fails that request only, never the identity for good.

        Failing closed must not become failing forever: the session that could
        not be opened is dropped, so the next request opens one of its own.
        """
        attempts = 0

        async def _assume_role(
            _session_name: str, _tag_value: str | None
        ) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            await sleep(0)
            if attempts == 1:
                throttled = make_client_error("ThrottlingException", "AssumeRole")
                raise throttled
            return {
                "access_key": _USER_KEY,
                "secret_key": "user-secret",
                "token": "user-token",
                "expiry_time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }

        monkeypatch.setattr(stdapi.aws, "_assume_user_role", _assume_role)
        with pytest.raises(ApiError):
            await user_role_credentials("alice")
        assert not stdapi.aws._USER_ROLE_CACHE  # noqa: SLF001
        credentials = await user_role_credentials("alice")
        assert (await credentials.get_frozen_credentials()).access_key == _USER_KEY
        assert attempts == 2


class TestRequestIdentity:
    """Which identity a request is attributed to, and what happens when it has none.

    Ref: stdapi/monitoring.py:resolve_request_identity
         stdapi/aws.py:request_user_role_credentials
    """

    async def test_authenticated_caller_outranks_the_declared_user(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """A verified caller is what the request is billed to, not what it declares."""
        calls = _stub_assume_role(monkeypatch)
        request_log["request_user_id"] = "declared-user"
        token = PRINCIPAL.set(Principal(subject="verified-user"))
        try:
            assert await stdapi.aws.request_user_role_credentials() is not None
        finally:
            PRINCIPAL.reset(token)
        assert calls[0][0].startswith("verified-user-")

    async def test_declared_user_is_used_without_a_verified_caller(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """An unauthenticated deployment attributes the request to what it declares."""
        calls = _stub_assume_role(monkeypatch)
        request_log["request_user_id"] = "declared-user"
        assert await stdapi.aws.request_user_role_credentials() is not None
        assert calls[0][0].startswith("declared-user-")

    async def test_session_is_recorded_in_the_request_log(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """The request log names the session the call was billed under.

        It is the only way to correlate a request with its Cost and Usage
        Report row, which carries the same session name.
        """
        _stub_assume_role(monkeypatch)
        request_log["request_user_id"] = "alice"
        await stdapi.aws.request_user_role_credentials()
        assert (
            request_log["aws_role_session_name"]
            == (user_role_session_identity("alice")[0])
        )

    async def test_request_without_identity_keeps_the_server_identity(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """By default a request naming no end user runs under the server's identity."""
        calls = _stub_assume_role(monkeypatch)
        assert await stdapi.aws.request_user_role_credentials() is None
        assert not calls

    async def test_request_without_identity_is_rejected_when_required(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """With the identity required, the request is rejected rather than mis-billed."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)
        _stub_assume_role(monkeypatch)
        with pytest.raises(ApiError) as raised:
            await stdapi.aws.request_user_role_credentials()
        assert raised.value.status == 400
        assert "safety_identifier" in str(raised.value)

    async def test_server_own_call_keeps_the_server_identity(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A call made outside any request is the server's own, whatever the policy.

        Startup and background work has no end user to attribute, and must not
        be blocked by a policy written about client requests.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)
        calls = _stub_assume_role(monkeypatch)
        assert await stdapi.aws.request_user_role_credentials() is None
        assert not calls

    async def test_feature_off_resolves_no_identity(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """With no role configured, nothing is resolved and no AWS STS call is made."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", None)
        calls = _stub_assume_role(monkeypatch)
        request_log["request_user_id"] = "alice"
        assert await stdapi.aws.request_user_role_credentials() is None
        assert not calls


class TestIdentityOffTheSigningPath:
    """The same requirement, for a transport botocore never signs.

    Bedrock Mantle sends plain HTTPS from the server's own credentials, so the
    signing hook above never runs on it. ``verify_user_role_identity`` is what
    keeps the documented ``400`` from depending on which endpoint happens to
    host the model; what it cannot restore is the role session itself, which is
    why routing a dual-homed model there is refused at startup.

    Ref: stdapi/aws.py:verify_user_role_identity
         stdapi/aws_bedrock_mantle.py:refuse_unattributable_invocation
    """

    @pytest.mark.usefixtures("user_role", "request_log")
    def test_a_request_without_identity_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal is the one the signing hook gives, status and text alike."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)

        with pytest.raises(ApiError) as raised:
            stdapi.aws.verify_user_role_identity()

        assert raised.value.status == 400
        assert "safety_identifier" in str(raised.value)

    @pytest.mark.usefixtures("user_role")
    def test_an_identified_request_passes(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """An end user the request names is all the requirement asks for."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)
        request_log["request_user_id"] = "alice"

        stdapi.aws.verify_user_role_identity()

    @pytest.mark.usefixtures("user_role")
    def test_a_server_own_call_is_not_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup and background work have no end user to name, and are not blocked."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)

        stdapi.aws.verify_user_role_identity()

    @pytest.mark.usefixtures("user_role", "request_log")
    def test_the_requirement_off_lets_the_request_through(self) -> None:
        """Without the requirement an unidentified request is served, as before."""
        stdapi.aws.verify_user_role_identity()


class TestSignedRequest:
    """The pooled client signs the invocation as the end user, not as the server.

    ``request_credentials`` is an internal botocore contract with no public
    guarantee, and it is what the whole feature rests on.

    Ref: stdapi/aws.py:_sign_model_invocation
    """

    class _AbortError(Exception):
        """Stops the request before it leaves the process."""

    async def _signed_headers(
        self, monkeypatch: pytest.MonkeyPatch, operation: str = "Converse"
    ) -> dict[str, Any]:
        """Return the headers of the request a Bedrock runtime call would send.

        botocore signs into raw header bytes, which its own type stubs declare
        as text; the assertions below read the bytes it really produces.
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
    async def test_invocation_is_signed_with_the_end_user_session(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
        operation: str,
    ) -> None:
        """The signature names the end user's session key, on the server's own client."""
        _stub_assume_role(monkeypatch)
        request_log["request_user_id"] = "alice"
        headers = await self._signed_headers(monkeypatch, operation)
        assert f"Credential={_USER_KEY}/".encode() in headers["Authorization"]
        assert _SERVER_KEY.encode() not in headers["Authorization"]
        assert headers["X-Amz-Security-Token"] == b"user-token"

    async def test_other_operations_keep_the_server_identity(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """Only model invocations run under the end user's role.

        Everything else the server calls on its own behalf would otherwise
        require the end user's role to carry those permissions too.
        """
        assert "ApplyGuardrail" not in USER_ROLE_OPERATIONS
        _stub_assume_role(monkeypatch)
        request_log["request_user_id"] = "alice"
        headers = await self._signed_headers(monkeypatch, "ApplyGuardrail")
        assert f"Credential={_SERVER_KEY}/".encode() in headers["Authorization"]

    async def test_two_users_are_signed_differently_on_one_client(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One pooled client signs each request as the user it is made for.

        Per-identity clients would mean a connection pool per end user; the
        credentials travel with the request instead.
        """
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        keys = {}
        for user in ("alice", "bob"):
            _stub_assume_role(monkeypatch)
            monkeypatch.setattr(
                stdapi.aws, "_assume_user_role", self._session_for(user)
            )
            token = REQUEST_LOG.set(make_event_log(request_user_id=user))
            try:
                headers = await self._signed_headers(monkeypatch)
            finally:
                REQUEST_LOG.reset(token)
            keys[user] = headers["X-Amz-Security-Token"]
        assert keys["alice"] != keys["bob"]

    @staticmethod
    def _session_for(
        user: str,
    ) -> Callable[[str, str | None], Awaitable[dict[str, Any]]]:
        """Return an assume-role stub whose session token names *user*."""

        async def _assume_role(_session_name: str, _tag: str | None) -> dict[str, Any]:
            return {
                "access_key": _USER_KEY,
                "secret_key": "user-secret",
                "token": f"token-{user}",
                "expiry_time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }

        return _assume_role

    async def test_identity_less_request_signs_with_the_server_identity(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """An invocation naming no end user is signed as the server, by default."""
        calls = _stub_assume_role(monkeypatch)
        headers = await self._signed_headers(monkeypatch)
        assert f"Credential={_SERVER_KEY}/".encode() in headers["Authorization"]
        assert not calls

    async def test_feature_off_signs_with_the_server_identity(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """With no role configured, requests are signed exactly as before."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", None)
        request_log["request_user_id"] = "alice"
        headers = await self._signed_headers(monkeypatch)
        assert f"Credential={_SERVER_KEY}/".encode() in headers["Authorization"]
        assert "X-Amz-Security-Token" not in headers

    async def test_failure_aborts_the_invocation(
        self,
        user_role: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """A request whose session cannot be opened is never sent to Bedrock."""
        _stub_assume_role(
            monkeypatch, error=make_client_error("AccessDenied", "AssumeRole")
        )
        request_log["request_user_id"] = "alice"
        with pytest.raises(ApiError):
            await self._signed_headers(monkeypatch)


class TestOperationScope:
    """The operations that run under the end user's role are the invocations.

    Ref: stdapi/aws.py:USER_ROLE_OPERATIONS
    """

    def test_covers_every_synchronous_invocation(self) -> None:
        """Both Converse APIs and both InvokeModel APIs are covered."""
        assert (
            frozenset(
                {
                    "Converse",
                    "ConverseStream",
                    "InvokeModel",
                    "InvokeModelWithResponseStream",
                }
            )
            == USER_ROLE_OPERATIONS
        )

    def test_every_operation_exists_in_the_wire_model(self) -> None:
        """Each name is a real Bedrock runtime operation, so none is a silent no-op."""
        operations = create_loader().load_service_model("bedrock-runtime", "service-2")[
            "operations"
        ]
        assert set(operations) >= USER_ROLE_OPERATIONS


class TestSettings:
    """A misconfigured per-end-user role is refused at startup, not at first request.

    Ref: stdapi/config.py:_validate_user_role_arn
    """

    @staticmethod
    def _settings(**values: Any) -> _Settings:  # noqa: ANN401
        """Build a settings instance carrying *values*."""
        return _Settings(**values)

    def test_role_arn_must_be_a_role_arn(self) -> None:
        """A value that is not an IAM role ARN fails validation."""
        with pytest.raises(ValueError, match="aws_bedrock_user_role_arn"):
            self._settings(aws_bedrock_user_role_arn="stdapi-ai-end-user")

    def test_role_arn_accepts_a_path_and_any_partition(self) -> None:
        """Role ARNs with a path, and non-commercial partitions, are accepted."""
        arn = "arn:aws-us-gov:iam::123456789012:role/team/stdapi-ai-end-user"
        assert self._settings(aws_bedrock_user_role_arn=arn)

    @pytest.mark.parametrize("duration", [899, 3601])
    def test_session_duration_is_clamped_to_what_sts_grants(
        self, duration: int
    ) -> None:
        """Outside 900-3600 seconds, AWS STS would reject every session.

        The upper bound is the one-hour ceiling AWS applies to a role session
        obtained from another role session.
        """
        with pytest.raises(ValueError, match="aws_bedrock_user_role_session_duration"):
            self._settings(
                aws_bedrock_user_role_arn=_ROLE_ARN,
                aws_bedrock_user_role_session_duration=duration,
            )

    @pytest.mark.parametrize("key", ["aws:user", "AWS:user", "bad*key", ""])
    def test_tag_key_must_be_one_sts_accepts(self, key: str) -> None:
        """A reserved or malformed tag key fails validation."""
        with pytest.raises(ValueError, match="aws_bedrock_user_role_tag_key"):
            self._settings(
                aws_bedrock_user_role_arn=_ROLE_ARN, aws_bedrock_user_role_tag_key=key
            )

    def test_requiring_an_identity_needs_a_role(self) -> None:
        """Requiring an end user identity without a role attributes nothing."""
        with pytest.raises(ValueError, match="aws_bedrock_user_role_require_identity"):
            self._settings(aws_bedrock_user_role_require_identity=True)


class TestStartupCheck:
    """A role that cannot be assumed is reported at startup, not at first request.

    Ref: stdapi/aws.py:verify_user_role_access
    """

    @staticmethod
    def _no_delay(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Remove the propagation wait between two startup attempts, recording it."""
        delays: list[float] = []

        async def _sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(stdapi.aws, "sleep", _sleep)
        return delays

    async def test_disabled_feature_opens_no_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no role configured, startup makes no AWS STS call at all."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", None)
        calls = _stub_assume_role(monkeypatch)
        event = make_event_log(type="start")
        await stdapi.aws.verify_user_role_access(event)
        assert not calls
        assert "server_warnings" not in event

    async def test_assumable_role_reports_nothing(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A working configuration is silent, and is verified with its session tag.

        Tagging is a separate AWS action from assuming, so a check that sent no
        tag would pass against a trust policy every request then fails on.
        """
        calls = _stub_assume_role(monkeypatch)
        event = make_event_log(type="start")
        await stdapi.aws.verify_user_role_access(event)
        assert len(calls) == 1
        assert calls[0][1]
        assert "server_warnings" not in event

    async def test_propagation_delay_is_retried(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A role whose trust policy is still propagating is retried, then accepted.

        A task started right after the role is created would otherwise report a
        failure it recovers from seconds later.
        """
        self._no_delay(monkeypatch)
        attempts = 0

        async def _assume_role(
            _session_name: str, _tag_value: str | None
        ) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                denied = make_client_error("AccessDenied", "AssumeRole")
                raise denied
            return {
                "access_key": _USER_KEY,
                "secret_key": "user-secret",
                "token": "user-token",
                "expiry_time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }

        monkeypatch.setattr(stdapi.aws, "_assume_user_role", _assume_role)
        event = make_event_log(type="start")
        await stdapi.aws.verify_user_role_access(event)
        assert attempts == 2
        assert "server_warnings" not in event

    async def test_unassumable_role_warns_without_failing_startup(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A role that stays unassumable is reported, and the server still starts.

        Refusing to start would turn a slow IAM propagation into an outage,
        while each request that needs a session still fails closed on its own.

        The budget is asserted rather than "more than once": it is sized for
        the IAM propagation window, and a shorter one would report a failure
        every fresh deployment recovers from seconds later.
        """
        delays = self._no_delay(monkeypatch)
        calls = _stub_assume_role(
            monkeypatch, error=make_client_error("AccessDenied", "AssumeRole")
        )
        event = make_event_log(type="start")
        await stdapi.aws.verify_user_role_access(event)
        assert len(calls) == stdapi.aws._USER_ROLE_CHECK_ATTEMPTS  # noqa: SLF001
        assert delays == [stdapi.aws._USER_ROLE_CHECK_RETRY_DELAY] * (  # noqa: SLF001
            len(calls) - 1
        )
        warning = " ".join(str(w) for w in event["server_warnings"])
        assert "sts:TagSession" in warning


class TestAssumeRoleRequest:
    """The AssumeRole request is the one AWS STS documents.

    Ref: https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html
         stdapi/aws.py:_assume_user_role
    """

    @staticmethod
    def _stub_sts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Answer AssumeRole from a pooled client stub, recording its parameters."""
        sent: list[dict[str, Any]] = []

        class _Sts:
            @staticmethod
            async def assume_role(**params: Any) -> dict[str, Any]:  # noqa: ANN401
                sent.append(params)
                return {
                    "Credentials": {
                        "AccessKeyId": _USER_KEY,
                        "SecretAccessKey": "user-secret",
                        "SessionToken": "user-token",
                        "Expiration": datetime.now(UTC) + timedelta(hours=1),
                    }
                }

        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: _Sts()
        )
        return sent

    async def test_request_names_the_role_session_and_tag(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The role, the session name, the lifetime and the end user tag are sent."""
        sent = self._stub_sts(monkeypatch)
        credentials = await user_role_credentials("alice")
        session_name, tag_value = user_role_session_identity("alice")
        assert sent == [
            {
                "RoleArn": _ROLE_ARN,
                "RoleSessionName": session_name,
                "DurationSeconds": 3600,
                "Tags": [{"Key": "user", "Value": tag_value}],
            }
        ]
        assert set(sent[0]) <= set(_STS_SHAPES["AssumeRoleRequest"]["members"])
        frozen = await credentials.get_frozen_credentials()
        assert (frozen.access_key, frozen.token) == (_USER_KEY, "user-token")

    async def test_no_tag_key_sends_no_tag(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a tag key the session is untagged, needing no sts:TagSession.

        The end user is then distinguished by the session name alone, which is
        what the Cost and Usage Report reports either way.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_tag_key", None)
        sent = self._stub_sts(monkeypatch)
        await user_role_credentials("alice")
        assert "Tags" not in sent[0]


class TestPooledStsClient:
    """AWS STS is pooled at startup, and only where a role session can need it.

    The role session is opened from the pooled client rather than an ad-hoc
    one, so a missing pool entry is a failure of every attributed request.

    Ref: stdapi/aws.py:AWSConnectionManager
         stdapi/aws.py:_assume_user_role
    """

    @staticmethod
    def _stub_create_client(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
        """Stub client creation, recording every ``(service, region)`` warmed."""
        monkeypatch.setattr(stdapi.aws, "_CLIENTS", {})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        created: list[tuple[str, str]] = []

        class _FakeClientCM:
            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *_exc: object) -> None:
                return None

        def _fake_create_client(
            service: str,
            *,
            region_name: str,
            config: Any,  # noqa: ANN401, ARG001
        ) -> _FakeClientCM:
            created.append((service, region_name))
            return _FakeClientCM()

        monkeypatch.setattr(AWS_SESSION, "create_client", _fake_create_client)
        return created

    async def test_pool_warms_sts_when_a_role_is_configured(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The role session's client is warmed with the rest of the pool."""
        created = self._stub_create_client(monkeypatch)
        manager = stdapi.aws.AWSConnectionManager(("bedrock-runtime", None))
        await manager.__aenter__()
        try:
            assert ("sts", stdapi.aws._STS_REGION) in created  # noqa: SLF001
            assert stdapi.aws._STS_REGION in stdapi.aws._CLIENTS["sts"]  # noqa: SLF001
        finally:
            await manager.__aexit__(None, None, None)

    async def test_pool_warms_no_sts_without_a_role(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment without the feature pays for no extra client."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", None)
        created = self._stub_create_client(monkeypatch)
        manager = stdapi.aws.AWSConnectionManager(("bedrock-runtime", None))
        await manager.__aenter__()
        try:
            assert not [spec for spec in created if spec[0] == "sts"]
        finally:
            await manager.__aexit__(None, None, None)

    async def test_session_is_opened_from_the_pooled_client(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_assume_user_role`` reaches AWS STS through the pool, not around it.

        Every other test here stubs the call or the lookup; this one runs the
        real ``get_client``, whose miss is a ``KeyError`` and not the 503 the
        feature is designed to answer.
        """
        sent: list[dict[str, Any]] = []

        class _Sts:
            @staticmethod
            async def assume_role(**params: Any) -> dict[str, Any]:  # noqa: ANN401
                sent.append(params)
                return {
                    "Credentials": {
                        "AccessKeyId": _USER_KEY,
                        "SecretAccessKey": "user-secret",
                        "SessionToken": "user-token",
                        "Expiration": datetime.now(UTC) + timedelta(hours=1),
                    }
                }

        monkeypatch.setitem(
            stdapi.aws._CLIENTS,  # noqa: SLF001
            "sts",
            {stdapi.aws._STS_REGION: _Sts()},  # noqa: SLF001
        )
        credentials = await user_role_credentials("alice")
        assert (await credentials.get_frozen_credentials()).access_key == _USER_KEY
        assert sent[0]["RoleArn"] == _ROLE_ARN


class TestRealtimeScope:
    """A real-time model session cannot carry an end user, and is refused if it must.

    A bidirectional stream is signed once as it opens, from the server's own
    credentials: it has no per-request signing point, so a deployment that
    requires every model invocation to name its end user must refuse it rather
    than report its usage under the server.

    Ref: stdapi/aws.py:verify_bidi_user_role_policy
    """

    def test_realtime_model_session_is_refused_when_identity_is_required(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the identity required, the session is refused rather than mis-billed."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)
        with pytest.raises(ApiError) as raised:
            stdapi.aws.verify_bidi_user_role_policy("bedrock-runtime")
        assert raised.value.status == 400
        message = str(raised.value)
        for leak in ("arn:", "bedrock", "sts:", _ROLE_ARN):
            assert leak not in message

    def test_realtime_model_session_is_allowed_by_default(
        self, user_role: None
    ) -> None:
        """Without the requirement, the session runs under the server's identity."""
        stdapi.aws.verify_bidi_user_role_policy("bedrock-runtime")

    def test_other_services_are_never_refused(
        self, user_role: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only model invocations are in scope; speech synthesis is not one."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)
        stdapi.aws.verify_bidi_user_role_policy("polly")

    def test_feature_off_refuses_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no role configured, no session is in scope of the policy."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", None)
        stdapi.aws.verify_bidi_user_role_policy("bedrock-runtime")
