"""AWS account-identity discovery at server startup: ECS task metadata, then STS.

The account ID is needed to build ARNs and the server name is what identifies a
replica in the logs, so startup resolves both from the ECS task metadata endpoint
when running on ECS and falls back to ``sts:GetCallerIdentity`` otherwise.

Ref: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-metadata-endpoint-v4.html
     https://docs.aws.amazon.com/STS/latest/APIReference/API_GetCallerIdentity.html
     stdapi/aws.py:initialize_aws_account_info
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from stdapi import aws, server

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Task ARN returned by the fake ECS container metadata endpoint
_TASK_ARN = "arn:aws:ecs:eu-west-3:123456789012:task/cluster/abcdef0123456789"


@pytest.fixture(autouse=True)
def _restore_globals() -> Generator[None]:
    """Restore the module state the initialization mutates."""
    account_id = aws.AWS_ENVIRONMENT.get("account_id")
    server_name = server.SERVER_NAME
    yield
    server.SERVER_NAME = server_name
    if account_id is None:
        aws.AWS_ENVIRONMENT.pop("account_id", None)
    else:
        aws.AWS_ENVIRONMENT["account_id"] = account_id


@pytest.fixture
async def metadata_server(request: pytest.FixtureRequest) -> AsyncGenerator[TestServer]:
    """Fake ECS container metadata endpoint, closed on teardown.

    The indirect parameter is the number of container requests answered with a
    server error before the endpoint starts working; it defaults to none.

    Yields:
        The started test server.
    """
    remaining: int = getattr(request, "param", 0)

    async def container(_: web.Request) -> web.Response:
        nonlocal remaining
        if remaining:
            remaining -= 1
            raise web.HTTPInternalServerError
        return web.json_response({"Name": "main"})

    async def task(_: web.Request) -> web.Response:
        return web.json_response({"TaskARN": _TASK_ARN})

    app = web.Application()
    app.router.add_get("/v4/id", container)
    app.router.add_get("/v4/id/task", task)
    test_server = TestServer(app)
    await test_server.start_server()
    yield test_server
    await test_server.close()


@pytest.mark.parametrize(
    "metadata_server",
    [pytest.param(0, id="first-attempt"), pytest.param(2, id="after-two-failures")],
    indirect=True,
)
async def test_account_info_from_ecs_metadata(
    monkeypatch: pytest.MonkeyPatch, metadata_server: TestServer
) -> None:
    """The account ID and task-qualified server name come from the ECS task metadata.

    The account ID is field 4 of the task ARN and the task ID is the last segment
    of field 5; the container name comes from the container endpoint itself. The
    endpoint is answered by the ECS agent and can be slow or briefly unavailable
    during startup, so the two-failure case covers the retry path (up to
    ``_ECS_METADATA_ATTEMPTS`` attempts) still yielding no warning.

    Ref: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-metadata-endpoint-v4-fargate-response.html
         stdapi/aws.py:_set_account_info_from_ecs
    """
    monkeypatch.setattr(aws, "_ECS_METADATA_RETRY_DELAY", 0)
    monkeypatch.setenv(
        "ECS_CONTAINER_METADATA_URI_V4", str(metadata_server.make_url("/v4/id"))
    )

    assert await aws.initialize_aws_account_info() is None, (
        "a reachable metadata endpoint must not produce a startup warning"
    )

    assert aws.AWS_ENVIRONMENT["account_id"] == "123456789012"
    assert f"abcdef0123456789-main-{server.SERVER_ID}" == server.SERVER_NAME


async def test_ecs_metadata_ignores_the_proxy_environment(
    monkeypatch: pytest.MonkeyPatch, metadata_server: TestServer
) -> None:
    """The task metadata session never routes through a configured HTTP proxy.

    The endpoint is a link-local address served by the container agent on the
    task's own host, so it is unreachable through any proxy. Reading
    ``HTTP_PROXY`` here would send every metadata request to the proxy unless
    the operator also excluded the link-local address in ``NO_PROXY``, which
    would break proxied deployments that work today.

    Ref: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-metadata-endpoint-v4.html
         https://docs.aiohttp.org/en/stable/client_reference.html
         stdapi/aws.py:_set_account_info_from_ecs
    """
    sessions: list[ClientSession] = []

    def record(**kwargs: object) -> ClientSession:
        session = ClientSession(**kwargs)  # type: ignore[arg-type]
        sessions.append(session)
        return session

    monkeypatch.setattr(aws, "ClientSession", record)
    monkeypatch.setenv(
        "ECS_CONTAINER_METADATA_URI_V4", str(metadata_server.make_url("/v4/id"))
    )

    assert await aws.initialize_aws_account_info() is None

    assert sessions, "the metadata endpoint must be read over its own session"
    assert all(not session.trust_env for session in sessions)


async def test_unreachable_ecs_metadata_falls_back_to_sts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable metadata endpoint warns and falls back to STS instead of failing.

    Startup must not depend on the ECS agent: the account ID still comes from STS,
    the returned warning names the exhausted attempt budget, and the server name is
    left un-qualified because no task ID could be read.

    Ref: stdapi/aws.py:_set_account_id_from_sts
    """
    monkeypatch.setattr(aws, "_ECS_METADATA_RETRY_DELAY", 0)
    # Port 1 on loopback is reserved and never bound, so the connection is refused
    # without racing another process for a just-released ephemeral port.
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://127.0.0.1:1/v4/id")
    called = False

    async def sts() -> None:
        nonlocal called
        called = True
        aws.AWS_ENVIRONMENT["account_id"] = "210987654321"

    monkeypatch.setattr(aws, "_set_account_id_from_sts", sts)
    server.SERVER_NAME = "unchanged"

    warning = await aws.initialize_aws_account_info()

    assert warning is not None
    assert "metadata endpoint unreachable" in warning
    assert f"after {aws._ECS_METADATA_ATTEMPTS} attempts" in warning  # noqa: SLF001
    assert "the server name does not identify the ECS task" in warning
    assert called, "STS is the mandatory fallback for the account ID"
    assert aws.AWS_ENVIRONMENT["account_id"] == "210987654321"
    assert server.SERVER_NAME == "unchanged"


async def test_account_info_from_sts_outside_ecs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the ECS metadata variable, the account ID comes from STS and no warning.

    Off ECS the metadata endpoint is legitimately absent, so this path is not a
    degraded mode: it returns no warning and leaves the server name unqualified.

    Ref: stdapi/aws.py:_set_account_id_from_sts
    """
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)

    async def sts() -> None:
        aws.AWS_ENVIRONMENT["account_id"] = "210987654321"

    monkeypatch.setattr(aws, "_set_account_id_from_sts", sts)
    server.SERVER_NAME = "unchanged"

    assert await aws.initialize_aws_account_info() is None
    assert aws.AWS_ENVIRONMENT["account_id"] == "210987654321"
    assert server.SERVER_NAME == "unchanged", (
        "no task metadata means no task-qualified server name"
    )
