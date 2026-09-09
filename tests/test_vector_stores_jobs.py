"""Which requests may hand their vector store indexing to the queue.

A queued job runs in whichever server receives it, outside any request, so
the embeddings it makes are signed as that server. A request whose model
invocations run under an identity of their own -- a tenant's registered AWS
credential, or a per-end-user role session -- would have that identity
silently replaced, and its spend moved onto the deployment's account. Those
waves stay in the server that accepted them, where the indexing task inherits
the request's identity, exactly as every wave does without a queue.
"""

from __future__ import annotations

from asyncio import Event, wait_for
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.config import SETTINGS
from stdapi.monitoring import (
    PRINCIPAL,
    TENANT,
    Principal,
    Tenant,
    TenantAwsCredential,
    tenant_aws_credential,
)
from stdapi.vector_stores import engine, jobs
from stdapi.vector_stores.engine import new_store_id

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.local

#: A well-formed queue URL, the one shape the setting accepts.
_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/stdapi-test-indexing"

#: A per-end-user role, as the setting accepts it.
_USER_ROLE_ARN = "arn:aws:iam::123456789012:role/stdapi-test-end-user"

#: One file identifier of the shape the Files API mints.
_FILE_ID = f"file-{0:032d}"

#: A tenant whose key carries an AWS credential of its own.
_TENANT_WITH_CREDENTIAL = Tenant(
    key_id="k" + "0" * 15,
    name="acme",
    aws_credential=TenantAwsCredential(
        role_arn="arn:aws:iam::210987654321:role/acme-stdapi", external_id="external-id"
    ),
)

#: The same tenant, having registered no credential.
_TENANT_WITHOUT_CREDENTIAL = Tenant(key_id="k" + "0" * 15, name="acme")


class _SendRecorder:
    """The one queue operation an attach uses, recording what it is given."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_message(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.sent.append(params)
        return {"MessageId": "message"}


@pytest.fixture
def indexing_queue(monkeypatch: pytest.MonkeyPatch) -> _SendRecorder:
    """Configure the indexing queue and record what is sent to it."""
    recorder = _SendRecorder()
    monkeypatch.setattr(SETTINGS, "aws_sqs_vector_store_queue_url", _QUEUE_URL)
    monkeypatch.setattr(jobs, "_client", lambda: recorder)
    return recorder


@pytest.fixture
def _no_request_identity() -> Iterator[None]:
    """Run the test as a request that carries neither a tenant nor a caller."""
    tenant_token = TENANT.set(None)
    principal_token = PRINCIPAL.set(None)
    try:
        yield
    finally:
        PRINCIPAL.reset(principal_token)
        TENANT.reset(tenant_token)


@pytest.fixture
def _tenant_credentials_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a tenant's registered credential sign its model invocations."""
    monkeypatch.setattr(SETTINGS, "tenant_api_keys", True)
    monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", True)


@pytest.mark.usefixtures("_no_request_identity")
class TestWhoseIdentityAQueuedJobWouldLose:
    """The hand-over is refused for a request the job could not sign as.

    Ref: stdapi/vector_stores/jobs.py:enqueue_indexing
         stdapi/vector_stores/jobs.py:_signs_as_the_request
         docs/operations_authentication_security.md (What runs on whose account)
    """

    @pytest.mark.usefixtures("_tenant_credentials_enabled")
    async def test_a_tenant_credential_keeps_the_wave_in_this_server(
        self, indexing_queue: _SendRecorder
    ) -> None:
        """A tenant paying for its own embeddings is never billed to the deployment.

        The queued job would sign as the server; the in-process task inherits
        the tenant and signs as its registered role.
        """
        TENANT.set(_TENANT_WITH_CREDENTIAL)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is False
        assert not indexing_queue.sent

    @pytest.mark.usefixtures("_tenant_credentials_enabled")
    async def test_a_tenant_without_a_credential_still_queues(
        self, indexing_queue: _SendRecorder
    ) -> None:
        """A tenant billed to the deployment loses nothing to the queue."""
        TENANT.set(_TENANT_WITHOUT_CREDENTIAL)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is True
        assert len(indexing_queue.sent) == 1

    async def test_a_tenant_credential_is_ignored_where_the_feature_is_off(
        self, indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With tenant credentials disabled the tenant signs as the server anyway."""
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", False)
        TENANT.set(_TENANT_WITH_CREDENTIAL)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is True
        assert len(indexing_queue.sent) == 1

    async def test_a_required_end_user_identity_keeps_the_wave_in_this_server(
        self, indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment refusing unattributed model usage cannot run the job.

        Outside a request the job identifies no end user, so every queued job
        would be refused the session it needs and fail its files.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", _USER_ROLE_ARN)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is False
        assert not indexing_queue.sent

    async def test_an_authenticated_caller_keeps_the_wave_in_this_server(
        self, indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller AWS reports separately keeps being reported separately."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", _USER_ROLE_ARN)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", False)
        PRINCIPAL.set(Principal(subject="user-1"))

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is False
        assert not indexing_queue.sent

    async def test_a_tenant_key_under_the_end_user_role_keeps_the_wave_here(
        self, indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tenant key is the end user the role session is opened for."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", _USER_ROLE_ARN)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", False)
        TENANT.set(_TENANT_WITHOUT_CREDENTIAL)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is False
        assert not indexing_queue.sent

    async def test_an_unattributed_request_under_the_end_user_role_queues(
        self, indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request the server already signs as itself loses nothing to the queue."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", _USER_ROLE_ARN)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", False)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is True
        assert len(indexing_queue.sent) == 1

    async def test_the_default_deployment_queues(
        self, indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither feature enabled: every wave goes to the queue, as before."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", None)
        monkeypatch.setattr(SETTINGS, "tenant_aws_credentials", False)

        assert await jobs.enqueue_indexing(new_store_id(), [_FILE_ID], "") is True
        assert len(indexing_queue.sent) == 1


@pytest.mark.usefixtures("_no_request_identity", "_tenant_credentials_enabled")
async def test_the_wave_kept_here_is_indexed_under_the_tenant(
    indexing_queue: _SendRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the refusal buys: the indexing task sees the tenant's credential.

    The attach hands the wave over from inside the request, so the task it
    starts instead of the send inherits the request's tenant -- which is what
    signs the embedding calls as the tenant's own role.

    Ref: stdapi/vector_stores/engine.py:_hand_over_indexing
         stdapi/aws.py:request_signing_credentials
    """
    seen: list[TenantAwsCredential | None] = []
    indexed = Event()

    async def _index_files(*_args: object) -> None:
        seen.append(tenant_aws_credential())
        indexed.set()

    monkeypatch.setattr(engine, "index_files", _index_files)
    TENANT.set(_TENANT_WITH_CREDENTIAL)
    store_id = new_store_id()

    await engine._hand_over_indexing(store_id, [_FILE_ID], "")  # noqa: SLF001
    await wait_for(indexed.wait(), timeout=5)

    assert not indexing_queue.sent
    assert seen == [_TENANT_WITH_CREDENTIAL.aws_credential]
