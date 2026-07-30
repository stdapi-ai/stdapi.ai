"""MCP server plumbing: request-ID propagation, structured error logging, transport.

An MCP tool call re-enters the gateway over an in-process ASGI transport, so it
would otherwise be logged as a second, unrelated request. The internal
user-agent plus a per-process internal header let the nested call reuse the
parent request ID, and ``fastapi_mcp``'s own logger is redirected into the
structured JSON log instead of printing tracebacks to stderr.

Ref: stdapi/mcp.py:mount_mcp
     stdapi/monitoring.py:log_request_event
"""

from __future__ import annotations

import logging
import sys
from contextvars import Context
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request as StarletteRequest

from stdapi import server
from stdapi.config import SETTINGS
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_LOG,
    EventLog,
    log_error_details,
    log_request_event,
)
from stdapi.utils import webuuid

if TYPE_CHECKING:
    from types import TracebackType

    from starlette.testclient import TestClient


def _make_request(
    method: str = "GET", path: str = "/test", headers: dict[str, str] | None = None
) -> StarletteRequest:
    """Build a minimal Starlette request for testing log_request_event."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
    }
    return StarletteRequest(scope)


def _capture_exc_info(
    msg: str,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """Raise and immediately catch a ValueError to produce a real exc_info tuple."""
    try:
        raise ValueError(msg)  # noqa: TRY301
    except ValueError:
        return sys.exc_info()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tests for request ID propagation (Issue 1)
# ---------------------------------------------------------------------------


class TestLogRequestEventIdPropagation:
    """log_request_event reuses the parent request ID only for internal MCP calls.

    Both the internal user-agent and the internal header must be present: the
    header name is randomised per process, and the user-agent check means a
    client cannot graft its request onto someone else's log entry.

    Ref: stdapi/monitoring.py:log_request_event
         stdapi/server.py:INTERNAL_REQUEST_ID_HEADER
    """

    def test_generates_new_id_for_external_requests(self) -> None:
        """A client-supplied request-ID header is ignored; a fresh ID is minted.

        ``x-stdapi-request-id`` is deliberately not the internal header name, so
        guessing the prefix is not enough to inject an ID.
        """
        request = _make_request(headers={"x-stdapi-request-id": "should-be-ignored"})
        with log_request_event(request) as log:
            assert log["id"] != "should-be-ignored"
            assert REQUEST_ID.get() == log["id"]

    def test_generates_new_id_when_only_header_present(self) -> None:
        """The internal header alone, without the MCP user-agent, does not reuse the ID."""
        request = _make_request(
            headers={server.INTERNAL_REQUEST_ID_HEADER: "parent-id-123"}
        )
        with log_request_event(request) as log:
            assert log["id"] != "parent-id-123"
            assert REQUEST_ID.get() == log["id"]

    def test_reuses_parent_id_for_internal_calls(self) -> None:
        """Internal calls (MCP user-agent + header) reuse the parent request ID.

        This is what collapses a tool call and the API request it triggers into a
        single log entry.
        """
        parent_id = webuuid()
        request = _make_request(
            headers={
                "user-agent": server.MCP_USER_AGENT,
                server.INTERNAL_REQUEST_ID_HEADER: parent_id,
            }
        )
        with log_request_event(request) as log:
            assert log["id"] == parent_id
            assert REQUEST_ID.get() == parent_id

    def test_request_id_context_var_is_reset_after_exit(self) -> None:
        """REQUEST_ID is restored to the outer value after log_request_event exits.

        Without the reset, a nested call would leave its own ID behind and the
        parent request's remaining log lines would be misattributed.
        """
        outer_id = "outer-request-id"
        outer_token = REQUEST_ID.set(outer_id)
        try:
            request = _make_request()
            with log_request_event(request) as log:
                inner_id = log["id"]
                assert inner_id != outer_id
                assert REQUEST_ID.get() == inner_id
            assert REQUEST_ID.get() == outer_id
        finally:
            REQUEST_ID.reset(outer_token)

    def test_request_id_reset_after_internal_call(self) -> None:
        """Parent request ID is restored after a nested internal call exits."""
        outer_id = webuuid()
        outer_token = REQUEST_ID.set(outer_id)
        try:
            inner_request = _make_request(
                headers={
                    "user-agent": server.MCP_USER_AGENT,
                    server.INTERNAL_REQUEST_ID_HEADER: outer_id,
                }
            )
            with log_request_event(inner_request):
                assert REQUEST_ID.get() == outer_id
            assert REQUEST_ID.get() == outer_id
        finally:
            REQUEST_ID.reset(outer_token)


# ---------------------------------------------------------------------------
# Tests for structured MCP error logging (Issue 2)
# ---------------------------------------------------------------------------


class TestMcpLogHandler:
    """_McpLogHandler routes fastapi_mcp errors into the structured JSON logger.

    ``fastapi_mcp`` logs tool failures through the stdlib logger, which would emit
    a raw traceback to stderr and bypass the JSON request log entirely.

    Ref: stdapi/mcp.py:_McpLogHandler
         stdapi/monitoring.py:log_error_details
    """

    @pytest.fixture(autouse=True)
    def _ensure_app_loaded(self, test_client: TestClient | None) -> None:
        """Ensure stdapi.main is imported (and the MCP handler registered) before each test."""
        if test_client is None:
            pytest.skip(
                "Requires local test server (MCP handler registered at app load)"
            )

    def _get_handler(self) -> logging.Handler:
        """Return the custom handler attached to the fastapi_mcp.server logger."""
        if not (SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse):
            pytest.skip("MCP is not enabled")
        mcp_logger = logging.getLogger("fastapi_mcp.server")
        handlers = [
            h for h in mcp_logger.handlers if type(h).__name__ == "_McpLogHandler"
        ]
        assert handlers, "No _McpLogHandler found on fastapi_mcp.server logger"
        return handlers[0]

    def test_handler_is_registered_and_no_propagation(self) -> None:
        """fastapi_mcp.server logger has the custom handler and propagate=False.

        ``propagate = False`` is what stops the root handler from also printing
        the record, so the message appears once, in the JSON log.
        """
        handler = self._get_handler()
        assert type(handler).__name__ == "_McpLogHandler"
        assert not logging.getLogger("fastapi_mcp.server").propagate

    def test_emit_adds_error_detail_to_request_log(self) -> None:
        """A record is appended to the active request's ``error_detail`` and raises its level.

        Promoting the entry to ``error`` is what makes the request show up as
        failed in the JSON log even though the HTTP response was a 200 JSON-RPC
        result carrying an in-band tool error.
        """
        handler = self._get_handler()

        log: EventLog = EventLog(
            type="request",
            level="info",
            date=MagicMock(),
            server_id="test-server",
            server_version="0.0.0",
        )
        log_token = REQUEST_LOG.set(log)
        try:
            record = logging.LogRecord(
                name="fastapi_mcp.server",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Error calling test_tool",
                args=(),
                exc_info=None,
            )
            handler.emit(record)
            assert "error_detail" in log
            assert any("Error calling test_tool" in str(d) for d in log["error_detail"])
            assert log["level"] == "error"
        finally:
            REQUEST_LOG.reset(log_token)

    def test_emit_includes_exception_traceback_in_error_detail(self) -> None:
        """A record carrying ``exc_info`` contributes the formatted traceback as a second detail.

        The traceback is the only thing that makes a tool failure diagnosable, so
        it is captured into the structured log rather than dropped with the record.
        """
        handler = self._get_handler()

        log: EventLog = EventLog(
            type="request",
            level="info",
            date=MagicMock(),
            server_id="test-server",
            server_version="0.0.0",
        )
        log_token = REQUEST_LOG.set(log)
        try:
            exc_info = _capture_exc_info("Underlying API returned 400")
            record = logging.LogRecord(
                name="fastapi_mcp.server",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Error calling search_models",
                args=(),
                exc_info=exc_info,
            )
            handler.emit(record)
            combined = " ".join(str(d) for d in log.get("error_detail", []))
            assert "Error calling search_models" in combined
            assert "ValueError" in combined
            assert "Underlying API returned 400" in combined
        finally:
            REQUEST_LOG.reset(log_token)

    def test_emit_is_safe_outside_request_context(self) -> None:
        """Outside a request context the record is dropped instead of raising LookupError.

        ``fastapi_mcp`` also logs during session setup and teardown, where no
        request log exists. Both calls run in a fresh, empty
        ``contextvars.Context`` so no ambient request log can hide the missing
        context, and the first one shows the error ``emit`` has to swallow.
        """
        handler = self._get_handler()
        record = logging.LogRecord(
            name="fastapi_mcp.server",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Unexpected error",
            args=(),
            exc_info=None,
        )
        with pytest.raises(LookupError):
            Context().run(log_error_details, "Unexpected error")
        assert Context().run(handler.emit, record) is None


# ---------------------------------------------------------------------------
# Integration tests: end-to-end MCP behaviour using the local test server
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    """End-to-end MCP behaviour over the streamable-HTTP transport mounted at /mcp.

    Ref: stdapi/mcp.py:mount_mcp
    """

    @pytest.fixture(scope="class")
    def client(self, test_client: TestClient | None) -> TestClient:
        """Return the session test client, skipping if not running locally."""
        if test_client is None:
            pytest.skip("Requires local test server")
        return test_client

    @pytest.fixture(scope="class")
    def mcp_session_id(self, client: TestClient, api_key: str) -> str:
        """Initialize an MCP session and return the session ID."""
        if not (SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse):
            pytest.skip("MCP is not enabled")
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert response.status_code == 200
        return response.headers["mcp-session-id"]  # type: ignore[no-any-return]

    def test_mcp_initialize_requires_authentication(self, client: TestClient) -> None:
        """An unauthenticated ``initialize`` is rejected with 401 and no session is created.

        The gateway's own ``authenticate`` dependency guards the transport, so the
        rejection happens before the MCP session manager runs and no
        ``mcp-session-id`` is handed out.

        Ref: stdapi/auth.py:authenticate
        """
        if not (SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse):
            pytest.skip("MCP is not enabled")
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert response.status_code == 401
        assert "mcp-session-id" not in response.headers

    def test_failing_tool_call_produces_no_traceback(
        self,
        client: TestClient,
        api_key: str,
        mcp_session_id: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A failing MCP tool call is answered in-band, with no Python traceback printed.

        JSON-RPC carries tool errors inside a 200 response, so the transport must
        still answer 200 while ``_McpLogHandler`` keeps the failure out of
        stdout/stderr and inside the structured log.

        Ref: stdapi/mcp.py:_McpLogHandler
        """
        capsys.readouterr()

        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 20,
                "method": "tools/call",
                "params": {
                    "name": "search_models",
                    "arguments": {"route": "/nonexistent/route"},
                },
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": mcp_session_id,
            },
        )
        captured = capsys.readouterr()
        assert response.status_code == 200
        assert "Traceback" not in captured.out
        assert "Traceback" not in captured.err

    def test_tool_descriptions_have_no_response_docs_block(
        self, client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """No tool description carries fastapi_mcp's auto-generated '### Responses:' section.

        That block restates the whole response schema for every tool and would
        dominate an MCP client's context budget, so it is stripped at mount time.

        Ref: stdapi/mcp.py:_strip_response_docs
        """
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 40, "method": "tools/list", "params": {}},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": mcp_session_id,
            },
        )
        assert response.status_code == 200
        tools = response.json()["result"]["tools"]
        assert tools
        assert all(
            "### Responses:" not in tool.get("description", "") for tool in tools
        )

    def test_mcp_response_has_request_id_header(
        self, client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """POST /mcp response carries the x-request-id header.

        /mcp is not tagged with a provider, so it falls back to the OpenAI
        convention ``x-request-id`` rather than Anthropic's ``request-id``.

        Ref: stdapi/api_providers/__init__.py:get_request_id_header
        """
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 30,
                "method": "tools/call",
                "params": {"name": "openai_model_list", "arguments": {}},
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": mcp_session_id,
            },
        )
        assert response.status_code == 200
        assert "x-request-id" in response.headers
