"""GET /health and GET /ping, the unauthenticated liveness probes.

Both are polled every few seconds for the life of a deployment, so both are
excluded from request logging: an orchestrator's probe traffic would otherwise
dominate the log and bill for its ingestion.

Ref: stdapi/routes/core_root.py:health_check
     stdapi/routes/core_root.py:ping
     stdapi/monitoring.py:LOGGING_PATHS_IGNORE
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.monitoring import LOGGING_PATHS_IGNORE

if TYPE_CHECKING:
    from starlette.testclient import TestClient

pytestmark = pytest.mark.local


class TestHealth:
    """GET /health liveness contract.

    Ref: stdapi/routes/core_root.py:health_check
         stdapi/routes/core_root.py:HealthResponse
    """

    def test_response_body(self, test_client: TestClient) -> None:
        """GET /health returns HTTP 200 with exactly ``{"status": "ok"}``.

        The route returns the ``HealthResponse`` dataclass whose only field
        defaults to ``"ok"``; FastAPI serialises it with no extra envelope.
        """
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_json_content_type(self, test_client: TestClient) -> None:
        """GET /health is served as ``application/json``."""
        response = test_client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_no_auth_required(self, test_client: TestClient) -> None:
        """GET /health succeeds with no credentials and with invalid ones alike.

        The route declares no ``Depends(authenticate)``, so an API key is
        neither required nor validated: a deliberately wrong bearer token must
        not turn into the 401 that authenticated routes return.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = test_client.get("/health")
        assert anonymous.status_code == 200
        assert anonymous.json() == {"status": "ok"}

        bad_key = test_client.get(
            "/health", headers={"Authorization": "Bearer wrong-key"}
        )
        assert bad_key.status_code == 200
        assert bad_key.json() == {"status": "ok"}


class TestPing:
    """GET /ping health contract, as Amazon Bedrock AgentCore Runtime defines it.

    AgentCore probes a hosted agent or MCP server at ``/ping`` and expects the
    exact body ``{"status": "Healthy"}``; a container that answers ``/health``
    only, or answers with any other body, is reported unhealthy and never
    receives traffic. The capitalised value is upstream's, not a typo.

    Ref: https://docs.aws.amazon.com/marketplace/latest/userguide/bedrock-agentcore-runtime.html
         stdapi/routes/core_root.py:ping
         stdapi/routes/core_root.py:PingResponse
    """

    def test_response_body(self, test_client: TestClient) -> None:
        """GET /ping returns HTTP 200 with exactly ``{"status": "Healthy"}``."""
        response = test_client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "Healthy"}

    def test_json_content_type(self, test_client: TestClient) -> None:
        """GET /ping is served as ``application/json``."""
        response = test_client.get("/ping")
        assert "application/json" in response.headers["content-type"]

    def test_no_auth_required(self, test_client: TestClient) -> None:
        """GET /ping succeeds with no credentials and with invalid ones alike.

        AgentCore's health probe carries no gateway API key, so requiring one
        would fail every deployment's readiness check.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = test_client.get("/ping")
        assert anonymous.status_code == 200

        bad_key = test_client.get(
            "/ping", headers={"Authorization": "Bearer wrong-key"}
        )
        assert bad_key.status_code == 200
        assert bad_key.json() == {"status": "Healthy"}


@pytest.mark.parametrize("path", ["/health", "/ping"])
class TestProbesAreNotLogged:
    """Neither probe produces a request log entry or a request ID.

    An orchestrator polls these every few seconds for the life of the
    deployment. Logging that traffic would bury real requests and bill for its
    ingestion, so the middleware skips its whole request-scoped block for these
    paths -- which is also why no ``x-request-id`` comes back.

    Ref: stdapi/main.py:_middleware
         stdapi/monitoring.py:LOGGING_PATHS_IGNORE
    """

    def test_path_is_excluded_from_logging(self, path: str) -> None:
        """The path is listed in the middleware's ignore set."""
        assert path in LOGGING_PATHS_IGNORE

    def test_no_request_log_is_emitted(
        self, test_client: TestClient, path: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Probing the path writes nothing to the structured JSON log on stdout."""
        capsys.readouterr()
        assert test_client.get(path).status_code == 200
        assert not capsys.readouterr().out.strip()

    def test_no_request_id_header(self, test_client: TestClient, path: str) -> None:
        """No request-ID header is returned, since no request event was opened.

        Ref: stdapi/api_providers/__init__.py:get_request_id_header
        """
        headers = test_client.get(path).headers
        assert "x-request-id" not in headers
        assert "request-id" not in headers

    def test_server_header_is_still_set(
        self, test_client: TestClient, path: str
    ) -> None:
        """The ``server`` header is set on the ignored path too.

        It is assigned after the branch, so a probe response stays identifiable
        even though nothing about it is logged.
        """
        assert test_client.get(path).headers["server"] == "stdapi.ai"
