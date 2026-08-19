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

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi_mcp import FastApiMCP  # type: ignore[import-untyped]
from starlette.requests import Request as StarletteRequest

from stdapi import server
from stdapi.config import SETTINGS
from stdapi.mcp import (
    _MCP_MAX_INLINE_BYTES,
    _bind_media_results,
    _lift_body_limit,
    _make_stateless,
)
from stdapi.monitoring import REQUEST_ID, log_error_details, log_request_event
from stdapi.types.openai_audio import (
    AudioTranscriptionJsonBody,
    AudioTranslationJsonBody,
)
from stdapi.utils import webuuid

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType
    from typing import Any

    from starlette.testclient import TestClient

#: Whether at least one MCP transport is enabled, which gates every /mcp test.
_MCP_ENABLED = SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse


def _mcp_post(
    client: TestClient,
    api_key: str,
    method: str,
    params: dict[str, Any],
    *,
    request_id: int,
    session_id: str | None = None,
    authenticated: bool = True,
) -> Any:  # noqa: ANN401
    """POST one JSON-RPC envelope to the streamable-HTTP transport.

    ``Accept`` must list both media types: the transport answers either JSON or
    an SSE stream depending on the method.

    Args:
        client: Test client bound to the local server.
        api_key: Bearer token for the gateway's own authentication.
        method: JSON-RPC method name.
        params: JSON-RPC params object.
        request_id: JSON-RPC request id.
        session_id: MCP session ID, for every call after ``initialize``.
        authenticated: Send the ``Authorization`` header.

    Returns:
        The raw HTTP response.
    """
    headers = {"Accept": "application/json, text/event-stream"}
    if authenticated:
        headers["Authorization"] = f"Bearer {api_key}"
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        headers=headers,
    )


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


@pytest.mark.local
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


@pytest.mark.local
@pytest.mark.skipif(not _MCP_ENABLED, reason="MCP is not enabled")
@pytest.mark.usefixtures("test_client")
class TestMcpLogHandler:
    """_McpLogHandler routes fastapi_mcp errors into the structured JSON logger.

    ``fastapi_mcp`` logs tool failures through the stdlib logger, which would emit
    a raw traceback to stderr and bypass the JSON request log entirely.

    Ref: stdapi/mcp.py:_McpLogHandler
         stdapi/monitoring.py:log_error_details
    """

    def _get_handler(self) -> logging.Handler:
        """Return the custom handler attached to the fastapi_mcp.server logger."""
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

    def test_emit_adds_error_detail_to_request_log(
        self, request_log: dict[str, Any]
    ) -> None:
        """A record is appended to the active request's ``error_detail`` and raises its level.

        Promoting the entry to ``error`` is what makes the request show up as
        failed in the JSON log even though the HTTP response was a 200 JSON-RPC
        result carrying an in-band tool error.
        """
        handler = self._get_handler()
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

        assert "error_detail" in request_log
        assert any(
            "Error calling test_tool" in str(detail)
            for detail in request_log["error_detail"]
        )
        assert request_log["level"] == "error"

    def test_emit_includes_exception_traceback_in_error_detail(
        self, request_log: dict[str, Any]
    ) -> None:
        """A record carrying ``exc_info`` contributes the formatted traceback as a second detail.

        The traceback is the only thing that makes a tool failure diagnosable, so
        it is captured into the structured log rather than dropped with the record.
        """
        handler = self._get_handler()
        record = logging.LogRecord(
            name="fastapi_mcp.server",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error calling search_models",
            args=(),
            exc_info=_capture_exc_info("Underlying API returned 400"),
        )

        handler.emit(record)

        combined = " ".join(
            str(detail) for detail in request_log.get("error_detail", [])
        )
        assert "Error calling search_models" in combined
        assert "ValueError" in combined
        assert "Underlying API returned 400" in combined

    def test_emit_is_safe_outside_request_context(self) -> None:
        """Outside a request context the record is dropped without raising.

        ``fastapi_mcp`` also logs during session setup and teardown, where no
        request log exists — as do the app's error handlers on log-exempt
        paths. Both calls run in a fresh, empty ``contextvars.Context`` so no
        ambient request log can hide the missing context.

        Ref: stdapi/monitoring.py:log_error_details
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
        assert Context().run(log_error_details, "Unexpected error") is None
        assert Context().run(handler.emit, record) is None


# ---------------------------------------------------------------------------
# Integration tests: end-to-end MCP behaviour using the local test server
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MCP_ENABLED, reason="MCP is not enabled")
class TestMCPIntegration:
    """End-to-end MCP behaviour over the streamable-HTTP transport mounted at /mcp.

    Ref: stdapi/mcp.py:mount_mcp
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def mcp_session_id(local_test_client: TestClient, api_key: str) -> str:
        """Initialize an MCP session and return the session ID."""
        response = _mcp_post(
            local_test_client,
            api_key,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
            request_id=1,
        )
        assert response.status_code == 200
        return response.headers["mcp-session-id"]  # type: ignore[no-any-return]

    def test_mcp_initialize_requires_authentication(
        self, local_test_client: TestClient, api_key: str
    ) -> None:
        """An unauthenticated ``initialize`` is rejected with 401 and no session is created.

        The gateway's own ``authenticate`` dependency guards the transport, so the
        rejection happens before the MCP session manager runs and no
        ``mcp-session-id`` is handed out.

        Ref: stdapi/auth.py:authenticate
        """
        response = _mcp_post(
            local_test_client,
            api_key,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
            request_id=1,
            authenticated=False,
        )
        assert response.status_code == 401
        assert "mcp-session-id" not in response.headers

    def test_failing_tool_call_produces_no_traceback(
        self,
        local_test_client: TestClient,
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

        response = _mcp_post(
            local_test_client,
            api_key,
            "tools/call",
            {"name": "search_models", "arguments": {"route": "/nonexistent/route"}},
            request_id=20,
            session_id=mcp_session_id,
        )
        captured = capsys.readouterr()
        assert response.status_code == 200
        assert "Traceback" not in captured.out
        assert "Traceback" not in captured.err

    def test_tool_descriptions_have_no_response_docs_block(
        self, local_test_client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """No tool description carries fastapi_mcp's auto-generated '### Responses:' section.

        That block restates the whole response schema for every tool and would
        dominate an MCP client's context budget, so it is stripped at mount time.

        Ref: stdapi/mcp.py:_strip_response_docs
        """
        response = _mcp_post(
            local_test_client,
            api_key,
            "tools/list",
            {},
            request_id=40,
            session_id=mcp_session_id,
        )
        assert response.status_code == 200
        tools = response.json()["result"]["tools"]
        assert tools
        assert all(
            "### Responses:" not in tool.get("description", "") for tool in tools
        )

    def test_mcp_response_has_request_id_header(
        self, local_test_client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """POST /mcp response carries the x-request-id header.

        /mcp is not tagged with a provider, so it falls back to the OpenAI
        convention ``x-request-id`` rather than Anthropic's ``request-id``.

        Ref: stdapi/api_providers/__init__.py:get_request_id_header
        """
        response = _mcp_post(
            local_test_client,
            api_key,
            "tools/call",
            {"name": "openai_model_list", "arguments": {}},
            request_id=30,
            session_id=mcp_session_id,
        )
        assert response.status_code == 200
        assert "x-request-id" in response.headers

    def test_tool_schemas_are_curated(
        self, local_test_client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """Served tool schemas carry no hidden params and no union/type conflicts.

        Ref: stdapi/mcp.py:_prune_hidden_params
             stdapi/mcp.py:_fix_union_param_types
        """
        from stdapi.mcp import _HIDDEN_TOOL_PARAMS  # noqa: PLC0415

        response = _mcp_post(
            local_test_client,
            api_key,
            "tools/list",
            {},
            request_id=41,
            session_id=mcp_session_id,
        )
        assert response.status_code == 200
        tools = response.json()["result"]["tools"]
        assert tools
        for tool in tools:
            properties = tool["inputSchema"].get("properties") or {}
            assert not _HIDDEN_TOOL_PARAMS.intersection(properties), tool["name"]
            for name, prop in properties.items():
                if isinstance(prop, dict) and "anyOf" in prop:
                    assert "type" not in prop, (tool["name"], name)

    @pytest.mark.local
    def test_audio_tool_schemas_are_usable_over_json(
        self, local_test_client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """The audio tools advertise only parameters their JSON body model accepts.

        A tool call always reaches the routes as ``application/json``, validated
        by ``AudioTranscriptionJsonBody``/``AudioTranslationJsonBody``, while the
        advertised schema is derived from the routes' multipart form fields. Any
        property those models do not have — such as the multipart-only ``name[]``
        aliases — is a parameter no MCP client can successfully send, and
        ``timestamp_granularities`` must be advertised as the list they require.

        Ref: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
             stdapi/routes/openai_audio_transcriptions.py:create_transcription
        """
        response = _mcp_post(
            local_test_client,
            api_key,
            "tools/list",
            {},
            request_id=43,
            session_id=mcp_session_id,
        )
        assert response.status_code == 200
        tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}

        for name, model in (
            ("openai_audio_transcription", AudioTranscriptionJsonBody),
            ("openai_audio_translation", AudioTranslationJsonBody),
        ):
            properties = tools[name]["inputSchema"]["properties"]
            assert not set(properties) - set(model.model_fields), name

        granularities = tools["openai_audio_transcription"]["inputSchema"][
            "properties"
        ]["timestamp_granularities"]
        assert all(
            option.get("type") in {"array", "null"}
            for option in granularities.get("anyOf", [granularities])
        )

    def test_tool_result_is_compact_json(
        self, local_test_client: TestClient, api_key: str, mcp_session_id: str
    ) -> None:
        """A JSON tool result is rendered compactly, not with indent=2.

        Re-encoding the parsed payload with compact separators must reproduce
        the served text exactly, proving no indentation whitespace was added.

        Ref: stdapi/mcp.py:_CompactJson
        """
        import json  # noqa: PLC0415

        response = _mcp_post(
            local_test_client,
            api_key,
            "tools/call",
            {"name": "openai_model_list", "arguments": {}},
            request_id=42,
            session_id=mcp_session_id,
        )
        assert response.status_code == 200
        for line in response.text.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                break
        else:
            payload = response.json()
        text = payload["result"]["content"][0]["text"]
        compact = json.dumps(
            json.loads(text), separators=(",", ":"), ensure_ascii=False
        )
        assert text == compact


def _mcp_only_app(*, stateless: bool) -> FastAPI:
    """Build a throwaway app exposing one tool over the streamable-HTTP transport.

    The gateway's own app mounts MCP once per session from the settings, so the
    two transport modes cannot both be observed on it. This isolates the mount
    itself, which is all the mode changes.

    Args:
        stateless: Whether to apply :func:`~stdapi.mcp._make_stateless`.

    Returns:
        The app, ready to serve ``/mcp``.
    """
    app = FastAPI()

    @app.get("/echo", operation_id="echo")
    async def echo() -> dict[str, str]:
        """Return a fixed payload."""
        return {"echo": "ok"}

    mcp = FastApiMCP(app, name="test", description="test")
    mcp.mount_http()
    if stateless:
        _make_stateless(mcp)
    return app


@pytest.mark.local
class TestStreamableHttpBodyLimit:
    """The transport must not refuse media the same tool accepts over HTTP.

    The MCP SDK caps a request body at 4 MiB and answers 413 before parsing,
    which is below what the image tools carry once base64 has inflated it. The
    mount raises that ceiling to the API's own limit.

    Ref: stdapi/mcp.py:_lift_body_limit
         https://github.com/stdapi-ai/stdapi.ai/issues/140
    """

    @staticmethod
    def _mounted_limit(*, lift: bool) -> int:
        """Mount the transport, start it, and read the body limit it enforces.

        Args:
            lift: Whether to apply :func:`~stdapi.mcp._lift_body_limit`.

        Returns:
            The ceiling in bytes the transport rejects a body above.
        """
        from starlette.testclient import TestClient  # noqa: PLC0415

        app = FastAPI()

        @app.get("/echo", operation_id="echo")
        async def echo() -> dict[str, str]:
            """Return a fixed payload."""
            return {"echo": "ok"}

        mcp = FastApiMCP(app, name="test", description="test")
        mcp.mount_http()
        if lift:
            _lift_body_limit(mcp)
        with TestClient(app) as client:
            client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers={"Accept": "application/json, text/event-stream"},
            )
            manager = mcp._http_transport._session_manager  # noqa: SLF001
            return int(manager.asgi_app.max_body_size)

    def test_the_sdk_default_is_below_what_an_image_tool_carries(self) -> None:
        """Without the lift, the transport refuses a modest base64 image.

        This is what shipped: the ceiling is low enough that an agent could not
        edit an image it could edit through the same tool over HTTP.
        """
        assert self._mounted_limit(lift=False) < 8 * 1024 * 1024

    def test_the_mount_raises_the_ceiling(self) -> None:
        """The mounted transport accepts what the API itself accepts."""
        assert self._mounted_limit(lift=True) >= 64 * 1024 * 1024

    def test_a_configured_input_limit_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator who bounds input files bounds the transport with it."""
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 5 * 1024 * 1024)
        assert self._mounted_limit(lift=True) == 5 * 1024 * 1024


def _tools_list(
    app: FastAPI, *, session_id: str | None, api_key: str | None = None
) -> Any:  # noqa: ANN401
    """POST a bare ``tools/list`` to *app*, without initializing a session first.

    Args:
        app: App exposing the transport.
        session_id: Value for the ``Mcp-Session-Id`` header, or None to omit it.
        api_key: Bearer token, for a transport guarded by ``authenticate``.

    Returns:
        The raw HTTP response.
    """
    from starlette.testclient import TestClient  # noqa: PLC0415

    headers = {"Accept": "application/json, text/event-stream"}
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    with TestClient(app) as client:
        return client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers=headers,
        )


@pytest.mark.local
class TestStatelessStreamableHttp:
    """``mcp_stateless_http`` makes /mcp answer a request that owns no session.

    Amazon Bedrock AgentCore Runtime provides its own session isolation and
    stamps an ``Mcp-Session-Id`` header on every request that lacks one, so the
    server sees an ID it never issued and no ``initialize`` handshake of its own.
    ``fastapi_mcp`` hard-codes the session manager to stateful, where that is a
    rejected request; the stateless mode AgentCore requires is what this setting
    turns on.

    Ref: https://docs.aws.amazon.com/marketplace/latest/userguide/bedrock-agentcore-runtime.html
         stdapi/mcp.py:_make_stateless
         stdapi/config.py:_Settings.mcp_stateless_http
    """

    def test_unknown_session_id_is_served(self) -> None:
        """A ``tools/list`` bearing a session ID the server never issued succeeds.

        This is the exact shape AgentCore forwards, and the reason the default
        transport cannot be deployed there unchanged.
        """
        response = _tools_list(
            _mcp_only_app(stateless=True), session_id="agentcore-injected-id"
        )
        assert response.status_code == 200
        names = [tool["name"] for tool in response.json()["result"]["tools"]]
        assert names == ["echo"]

    def test_no_session_id_is_served(self) -> None:
        """A ``tools/list`` with no session header at all succeeds too.

        Stateless mode has no handshake to skip, so the first request a client
        makes is answered whether or not the host stamped an ID on it.
        """
        response = _tools_list(_mcp_only_app(stateless=True), session_id=None)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("session_id", "expected_status"), [(None, 400), ("agentcore-injected-id", 404)]
    )
    def test_the_default_transport_rejects_the_same_request(
        self, session_id: str | None, expected_status: int
    ) -> None:
        """Without the setting, the same request is refused rather than served.

        The negative control: it is the mode, not the payload, that decides. A
        missing session ID is a bad request; an unrecognised one is a session
        the stateful manager never issued.
        """
        response = _tools_list(_mcp_only_app(stateless=False), session_id=session_id)
        assert response.status_code == expected_status


#: Operation IDs of the throwaway app used to observe include/exclude filtering.
_TOOL_OPERATION_IDS = ("tool_alpha", "tool_beta", "tool_gamma")


def _three_tool_app() -> FastAPI:
    """Build an app exposing one operation per :data:`_TOOL_OPERATION_IDS` entry."""
    app = FastAPI()
    for operation_id in _TOOL_OPERATION_IDS:

        async def endpoint() -> dict[str, str]:
            """Return a fixed payload."""
            return {"ok": "yes"}

        app.get(f"/{operation_id}", operation_id=operation_id)(endpoint)
    return app


@pytest.mark.local
class TestToolSelectionSettings:
    """MCP_INCLUDE_TOOLS / MCP_EXCLUDE_TOOLS decide which tools the server advertises.

    What an operator buys with either setting is an operation that no MCP client
    can see or call, so the mount is driven with each filter in turn and both the
    ``tools/list`` answer and the SEP-1649 server card are checked, since a
    client may read either. The settings are applied here directly, past the
    include/exclude reconciliation :class:`~stdapi.config._Settings` performs at
    validation time.

    Ref: stdapi/mcp.py:mount_mcp
         stdapi/config.py:_Settings.mcp_include_tools
         stdapi/routes/core_root.py:MCP_SERVER_CARD
    """

    @staticmethod
    def _advertised(
        monkeypatch: pytest.MonkeyPatch,
        api_key: str,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Mount MCP on a throwaway app under the given filters.

        Args:
            monkeypatch: Fixture used to scope the settings and the server card.
            api_key: Bearer token for the ``authenticate`` dependency the mount adds.
            include: Value for ``mcp_include_tools``.
            exclude: Value for ``mcp_exclude_tools``.

        Returns:
            The tool names ``tools/list`` returned, and the ones on the server card.
        """
        import logging as stdlib_logging  # noqa: PLC0415

        from stdapi import mcp as stdapi_mcp  # noqa: PLC0415

        card: dict[str, Any] = {}
        monkeypatch.setattr(SETTINGS, "mcp_include_tools", include)
        monkeypatch.setattr(SETTINGS, "mcp_exclude_tools", exclude)
        monkeypatch.setattr(SETTINGS, "enable_mcp_streamable_http", True)
        monkeypatch.setattr(SETTINGS, "enable_mcp_sse", False)
        monkeypatch.setattr(SETTINGS, "mcp_stateless_http", True)
        monkeypatch.setattr(stdapi_mcp, "MCP_SERVER_CARD", card)
        # Restored wholesale, so the extra log handler mount_mcp installs on the
        # shared fastapi_mcp logger does not leak into the rest of the session.
        mcp_logger = stdlib_logging.getLogger("fastapi_mcp.server")
        monkeypatch.setattr(mcp_logger, "handlers", list(mcp_logger.handlers))

        app = _three_tool_app()
        stdapi_mcp.mount_mcp(app)
        response = _tools_list(app, session_id=None, api_key=api_key)
        assert response.status_code == 200
        served = [tool["name"] for tool in response.json()["result"]["tools"]]
        return served, [tool["name"] for tool in card["tools"]]

    @pytest.mark.usefixtures("local_test_client")
    def test_excluded_operation_is_not_advertised(
        self, monkeypatch: pytest.MonkeyPatch, api_key: str
    ) -> None:
        """An excluded operation ID disappears from tools/list and the server card.

        The other two operations stay, so the exclusion is what removed it
        rather than the mount failing to discover anything.
        """
        served, card = self._advertised(monkeypatch, api_key, exclude=["tool_beta"])
        assert served == ["tool_alpha", "tool_gamma"]
        assert card == ["tool_alpha", "tool_gamma"]

    @pytest.mark.usefixtures("local_test_client")
    def test_include_list_is_the_whole_advertised_set(
        self, monkeypatch: pytest.MonkeyPatch, api_key: str
    ) -> None:
        """With an include list, only the named operations are advertised."""
        served, card = self._advertised(monkeypatch, api_key, include=["tool_gamma"])
        assert served == ["tool_gamma"]
        assert card == ["tool_gamma"]

    @pytest.mark.usefixtures("local_test_client")
    def test_unfiltered_advertises_every_operation(
        self, monkeypatch: pytest.MonkeyPatch, api_key: str
    ) -> None:
        """Unset, both settings advertise the app's full operation set.

        The negative control for the two filtering tests above.
        """
        served, card = self._advertised(monkeypatch, api_key)
        assert served == list(_TOOL_OPERATION_IDS)
        assert card == list(_TOOL_OPERATION_IDS)


# ---------------------------------------------------------------------------
# Tool schema curation: hidden params, union type repair, compact results
# ---------------------------------------------------------------------------


def _make_tool(schema: dict[str, Any]) -> Any:  # noqa: ANN401
    """Build a minimal MCP Tool carrying *schema* as its input schema."""
    from mcp.types import Tool  # noqa: PLC0415

    return Tool(name="tool", inputSchema=schema)


@pytest.mark.local
class TestPruneHiddenParams:
    """_prune_hidden_params drops LLM-unusable params and orphaned $defs.

    The parameters stay accepted by the API routes; only their advertisement in
    the tool input schemas is removed, so they stop costing MCP client context.

    Ref: stdapi/mcp.py:_prune_hidden_params
    """

    def test_hidden_params_and_unreachable_defs_removed(self) -> None:
        """Hidden properties, their required entries, and now-orphaned $defs go away.

        ``Nested`` stays because the surviving ``prompt`` still reaches it;
        ``StreamOptions`` was only reachable through the removed ``stream``.
        """
        from stdapi.mcp import _prune_hidden_params  # noqa: PLC0415

        tool = _make_tool(
            {
                "type": "object",
                "properties": {
                    "stream": {"$ref": "#/$defs/StreamOptions"},
                    "prompt": {
                        "anyOf": [{"type": "string"}, {"$ref": "#/$defs/Nested"}]
                    },
                },
                "required": ["prompt", "stream"],
                "$defs": {
                    "StreamOptions": {"type": "object"},
                    "Nested": {"items": {"type": "string"}, "type": "array"},
                },
            }
        )
        _prune_hidden_params([tool])
        assert "stream" not in tool.inputSchema["properties"]
        assert tool.inputSchema["required"] == ["prompt"]
        assert "StreamOptions" not in tool.inputSchema["$defs"]
        assert "Nested" in tool.inputSchema["$defs"]

    def test_tool_without_hidden_params_untouched(self) -> None:
        """A schema with no hidden parameter is left byte-identical."""
        from copy import deepcopy  # noqa: PLC0415

        from stdapi.mcp import _prune_hidden_params  # noqa: PLC0415

        schema = {
            "type": "object",
            "properties": {"model": {"type": "string"}},
            "required": ["model"],
        }
        tool = _make_tool(deepcopy(schema))
        _prune_hidden_params([tool])
        assert tool.inputSchema == schema


@pytest.mark.local
class TestFixUnionParamTypes:
    """_fix_union_param_types removes the random ``type`` beside ``anyOf``.

    ``fastapi_mcp`` derives that sibling ``type`` from an unordered set of the
    union member types, so a ``str | list`` parameter can advertise
    ``"type": "array"`` in one process and ``"string"`` in the next — and the
    MCP SDK's server-side jsonschema validation then rejects valid arguments.

    Ref: stdapi/mcp.py:_fix_union_param_types
         fastapi_mcp.openapi.utils.get_single_param_type_from_schema
    """

    def test_type_beside_anyof_is_removed(self) -> None:
        """The contradictory sibling ``type`` disappears; the anyOf stays."""
        from stdapi.mcp import _fix_union_param_types  # noqa: PLC0415

        tool = _make_tool(
            {
                "type": "object",
                "properties": {
                    "prompt": {
                        "anyOf": [{"type": "string"}, {"type": "array"}],
                        "type": "array",
                    }
                },
            }
        )
        _fix_union_param_types([tool])
        prompt = tool.inputSchema["properties"]["prompt"]
        assert "type" not in prompt
        assert prompt["anyOf"]

    def test_plain_typed_param_keeps_its_type(self) -> None:
        """A non-union parameter keeps its legitimate ``type``."""
        from stdapi.mcp import _fix_union_param_types  # noqa: PLC0415

        tool = _make_tool(
            {"type": "object", "properties": {"model": {"type": "string"}}}
        )
        _fix_union_param_types([tool])
        assert tool.inputSchema["properties"]["model"]["type"] == "string"

    def test_fastapi_mcp_still_injects_the_sibling_type(self) -> None:
        """The upstream helper still returns a single type for unions.

        If this fails after a ``fastapi_mcp`` upgrade, the union repair may have
        become unnecessary — re-evaluate :func:`_fix_union_param_types`.
        """
        from fastapi_mcp.openapi.utils import (  # type: ignore[import-untyped]  # noqa: PLC0415
            get_single_param_type_from_schema,
        )

        union = {"anyOf": [{"type": "string"}, {"type": "array"}]}
        assert get_single_param_type_from_schema(union) in {"string", "array"}


@pytest.mark.local
class TestCompactJson:
    """_CompactJson renders tool results compactly in place of stdlib indent=2.

    Ref: stdapi/mcp.py:_CompactJson
    """

    def test_dumps_is_compact_and_ignores_formatting_kwargs(self) -> None:
        """The indent/ensure_ascii kwargs fastapi_mcp passes are ignored."""
        from stdapi.mcp import _CompactJson  # noqa: PLC0415

        rendered = _CompactJson.dumps({"a": [1, 2], "é": "ü"}, indent=2)
        assert rendered == '{"a":[1,2],"é":"ü"}'

    def test_decode_error_is_the_stdlib_exception(self) -> None:
        """The façade exposes the exact exception class fastapi_mcp catches."""
        import json  # noqa: PLC0415

        from stdapi.mcp import _CompactJson  # noqa: PLC0415

        assert _CompactJson.JSONDecodeError is json.JSONDecodeError

    @pytest.mark.local
    @pytest.mark.skipif(not _MCP_ENABLED, reason="MCP is not enabled")
    @pytest.mark.usefixtures("test_client")
    def test_facade_installed_on_fastapi_mcp(self) -> None:
        """Mounting MCP swaps fastapi_mcp.server's ``json`` for the façade."""
        import fastapi_mcp.server  # type: ignore[import-untyped]  # noqa: PLC0415

        from stdapi.mcp import _CompactJson  # noqa: PLC0415

        assert fastapi_mcp.server.json is _CompactJson


# ---------------------------------------------------------------------------
# Routes answering with bytes rather than JSON
# ---------------------------------------------------------------------------


#: Bodies the throwaway media app serves, per operation ID: media type and payload.
_MEDIA_BODIES: dict[str, tuple[str, bytes]] = {
    # The leading null bytes are what json.loads reads as a UTF-32 signature.
    "media_video": ("video/mp4", b"\x00\x00\x00 ftypisom" + bytes(64)),
    "media_document": ("application/pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"),
    "media_image": ("image/png", b"\x89PNG\r\n\x1a\n" + bytes(32)),
    "media_audio": ("audio/mpeg", b"\xff\xfb\x90\x64" + bytes(32)),
    "media_huge_image": (
        "image/png",
        b"\x89PNG\r\n\x1a\n" + bytes(_MCP_MAX_INLINE_BYTES),
    ),
    "media_text": ("text/plain", b"hello mcp"),
}


def _media_app(*, bind: bool = True) -> FastAPI:
    """Build a throwaway app serving one body per :data:`_MEDIA_BODIES` entry.

    Args:
        bind: Whether to apply :func:`~stdapi.mcp._bind_media_results`, so the
            same calls can be observed with and without it.

    Returns:
        The app, ready to serve ``/mcp`` without a handshake.
    """
    app = FastAPI()

    def _route(payload: bytes, media_type: str) -> Any:  # noqa: ANN401
        """Build an endpoint streaming *payload* as *media_type*."""

        async def endpoint() -> StreamingResponse:
            """Stream the fixed body."""

            async def stream() -> AsyncIterator[bytes]:
                """Yield the body in one chunk."""
                yield payload

            return StreamingResponse(stream(), media_type=media_type)

        return endpoint

    for operation_id, (media_type, payload) in _MEDIA_BODIES.items():
        app.get(f"/{operation_id}", operation_id=operation_id)(
            _route(payload, media_type)
        )

    mcp = FastApiMCP(app, name="test", description="test")
    mcp.mount_http()
    _make_stateless(mcp)
    if bind:
        _bind_media_results(mcp)
    return app


def _call_tool(app: FastAPI, name: str) -> Any:  # noqa: ANN401
    """Call one tool of *app* over the streamable-HTTP transport.

    Args:
        app: App exposing the transport in stateless mode.
        name: Tool to call, with no arguments.

    Returns:
        The JSON-RPC ``result`` object, carrying ``content`` and ``isError``.
    """
    import json  # noqa: PLC0415

    from starlette.testclient import TestClient  # noqa: PLC0415

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": {}},
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code == 200, response.text
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])["result"]
    return response.json()["result"]


@pytest.mark.local
class TestBinaryToolResults:
    """A route answering with bytes must still be callable as an MCP tool.

    ``fastapi_mcp`` parses every response as JSON and falls back to its decoded
    text, neither of which a binary body survives, so every such tool was
    unusable: video content always, file content for anything that is not text.
    Media an MCP client can render travels as the content type built for it, and
    everything else is answered with a reference to its download URL rather than
    with bytes no agent could use.

    Ref: stdapi/mcp.py:_bind_media_results
         https://modelcontextprotocol.io/specification/2025-06-18/server/tools
    """

    @staticmethod
    def _reference(result: Any) -> Any:  # noqa: ANN401
        """Assert the call succeeded with a text block, and parse it as JSON."""
        import json  # noqa: PLC0415

        assert not result["isError"], result["content"]
        assert result["content"][0]["type"] == "text"
        return json.loads(result["content"][0]["text"])

    def test_a_video_body_answers_a_download_reference(self) -> None:
        """The video tool returns the URL to download from, not the MP4 bytes.

        No MCP content type carries video, and a generated clip is far larger
        than any agent's context, so the reference is the whole capability.
        """
        reference = self._reference(_call_tool(_media_app(), "media_video"))
        assert reference["content_type"] == "video/mp4"
        assert reference["url"] == "/media_video"

    def test_without_the_binding_the_same_call_fails(self) -> None:
        """The negative control: unbound, the call errors while decoding the body.

        This is what shipped — the MP4 header's null bytes are read as a UTF-32
        signature, and decoding them raises before any result is built.
        """
        result = _call_tool(_media_app(bind=False), "media_video")
        assert result["isError"]
        assert "decode" in result["content"][0]["text"]

    def test_a_non_text_file_body_answers_a_download_reference(self) -> None:
        """A binary file the client cannot render comes back as a reference too.

        The file-content tools serve whatever was uploaded, so this is the case
        an agent meets first: a PDF, an archive, anything not text.
        """
        reference = self._reference(_call_tool(_media_app(), "media_document"))
        assert reference["content_type"] == "application/pdf"
        assert reference["url"] == "/media_document"

    def test_an_image_body_is_returned_as_image_content(self) -> None:
        """An image small enough to travel reaches the agent as image content."""
        from base64 import b64decode  # noqa: PLC0415

        result = _call_tool(_media_app(), "media_image")
        assert not result["isError"], result["content"]
        block = result["content"][0]
        assert block["type"] == "image"
        assert block["mimeType"] == "image/png"
        assert b64decode(block["data"]) == _MEDIA_BODIES["media_image"][1]

    def test_an_audio_body_is_returned_as_audio_content(self) -> None:
        """Generated speech reaches the agent as audio content, not as text."""
        from base64 import b64decode  # noqa: PLC0415

        result = _call_tool(_media_app(), "media_audio")
        assert not result["isError"], result["content"]
        block = result["content"][0]
        assert block["type"] == "audio"
        assert block["mimeType"] == "audio/mpeg"
        assert b64decode(block["data"]) == _MEDIA_BODIES["media_audio"][1]

    def test_media_above_the_inline_ceiling_answers_a_reference(self) -> None:
        """An image too large to inline degrades to a reference rather than failing.

        Base64 inflates a payload by a third, and an MCP client feeds a content
        block straight to its model, so the ceiling is what keeps a large asset
        from blowing up the very session that asked for it.
        """
        reference = self._reference(_call_tool(_media_app(), "media_huge_image"))
        assert reference["content_type"] == "image/png"

    def test_a_text_body_is_unchanged(self) -> None:
        """The control for every case above: text still arrives verbatim."""
        result = _call_tool(_media_app(), "media_text")
        assert not result["isError"], result["content"]
        assert result["content"][0]["text"] == "hello mcp"
