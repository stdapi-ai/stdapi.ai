"""Offline tests for AWS request-ID capture and edge correlation headers.

The botocore ``after-call``/``after-call-error`` hooks are exercised with
stubbed parsed responses (no AWS API is ever called), and the edge correlation
headers with the in-process ASGI client.

Ref: stdapi/aws.py:_record_after_call
     stdapi/aws.py:_record_after_call_error
     stdapi/monitoring.py:record_aws_api_call
     stdapi/monitoring.py:log_request_event
"""

from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ClientError
from starlette.requests import Request as StarletteRequest

from stdapi import monitoring, usage
from stdapi.aws import _record_after_call, _record_after_call_error
from stdapi.config import AWS_SESSION
from stdapi.monitoring import (
    _AWS_API_CALLS,
    _AWS_REQUESTS_MAX,
    _EDGE_HEADER_MAX_LENGTH,
    EventLog,
    log_background_event,
    log_request_event,
    log_request_stream_event,
    record_aws_api_call,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from botocore.model import OperationModel
    from starlette.testclient import TestClient

    from stdapi.monitoring import AwsApiCallLog

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _make_request(headers: list[tuple[bytes, bytes]] | None = None) -> StarletteRequest:
    """Build a minimal Starlette request carrying *headers*."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "query_string": b"",
        "headers": headers or [],
    }
    return StarletteRequest(scope)


def _make_client_error_with_request_id(code: str, request_id: str) -> ClientError:
    """Build a ClientError whose response carries an AWS request ID."""
    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"RequestId": request_id},
        },
        "PutObject",
    )


@pytest.fixture
def aws_calls() -> Generator[list[AwsApiCallLog]]:
    """Provide a request-scoped AWS API call accumulator, reset on exit."""
    token = _AWS_API_CALLS.set([])
    yield _AWS_API_CALLS.get()
    _AWS_API_CALLS.reset(token)


@pytest.fixture(scope="module")
def put_object_model() -> OperationModel:
    """Real S3 PutObject operation model, loaded from the local botocore data."""
    from botocore.session import get_session  # noqa: PLC0415

    return get_session().get_service_model("s3").operation_model("PutObject")


class TestAfterCallHooks:
    """The botocore hook handlers, driven with stubbed parsed responses."""

    def test_after_call_records_the_request_id(
        self, aws_calls: list[AwsApiCallLog], put_object_model: OperationModel
    ) -> None:
        """A successful call records service, operation and RequestId.

        Ref: stdapi/aws.py:_record_after_call
        """
        _record_after_call(
            parsed={"ResponseMetadata": {"RequestId": "req-ok"}},
            model=put_object_model,
            context={},
        )
        assert aws_calls == [
            {"service": "s3", "operation": "PutObject", "request_id": "req-ok"}
        ]

    def test_after_call_records_the_error_code_of_failed_calls(
        self, aws_calls: list[AwsApiCallLog], put_object_model: OperationModel
    ) -> None:
        """An HTTP error response records its request ID with the AWS error code.

        botocore emits ``after-call`` for error responses too, before raising
        the matching ClientError.

        Ref: stdapi/aws.py:_record_after_call
        """
        _record_after_call(
            parsed={
                "Error": {"Code": "ThrottlingException", "Message": "slow down"},
                "ResponseMetadata": {"RequestId": "req-throttled"},
            },
            model=put_object_model,
            context={},
        )
        assert aws_calls == [
            {
                "service": "s3",
                "operation": "PutObject",
                "request_id": "req-throttled",
                "error": "ThrottlingException",
            }
        ]

    def test_after_call_falls_back_to_the_requestid_header(
        self, aws_calls: list[AwsApiCallLog], put_object_model: OperationModel
    ) -> None:
        """Without ResponseMetadata.RequestId, the x-amzn-requestid header is used.

        Ref: stdapi/aws.py:_aws_request_id
        """
        _record_after_call(
            parsed={
                "ResponseMetadata": {"HTTPHeaders": {"x-amzn-requestid": "req-header"}}
            },
            model=put_object_model,
            context={},
        )
        assert aws_calls[0]["request_id"] == "req-header"

    def test_after_call_without_request_id_records_nothing(
        self, aws_calls: list[AwsApiCallLog], put_object_model: OperationModel
    ) -> None:
        """A response with no request ID at all produces no entry.

        Ref: stdapi/aws.py:_record_after_call
        """
        _record_after_call(parsed={}, model=put_object_model, context={})
        assert aws_calls == []

    def test_after_call_error_uses_the_exception_response(
        self, aws_calls: list[AwsApiCallLog]
    ) -> None:
        """A ClientError raised before the after-call stage still gets recorded.

        Ref: stdapi/aws.py:_record_after_call_error
        """
        _record_after_call_error(
            event_name="after-call-error.s3.PutObject",
            exception=_make_client_error_with_request_id("AccessDenied", "req-err"),
            context={},
        )
        assert aws_calls == [
            {
                "service": "s3",
                "operation": "PutObject",
                "request_id": "req-err",
                "error": "AccessDenied",
            }
        ]

    def test_after_call_error_ignores_transport_errors(
        self, aws_calls: list[AwsApiCallLog]
    ) -> None:
        """An exception without a parsed AWS response (no request ID) is skipped.

        Ref: stdapi/aws.py:_record_after_call_error
        """
        _record_after_call_error(
            event_name="after-call-error.s3.PutObject",
            exception=ValueError("boom"),
            context={},
        )
        assert aws_calls == []

    async def test_hooks_fire_through_a_session_created_client(
        self, aws_calls: list[AwsApiCallLog]
    ) -> None:
        """Clients created from the shared session inherit both hooks.

        Emits the botocore events through a real client's event system (no AWS
        call is made) to prove the session-level registration propagates.

        Ref: stdapi/aws.py:AWS_SESSION.register
        """
        async with AWS_SESSION.create_client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",  # noqa: S106
        ) as client:
            await client.meta.events.emit(
                "after-call.s3.PutObject",
                http_response=None,
                parsed={"ResponseMetadata": {"RequestId": "req-emit"}},
                model=client.meta.service_model.operation_model("PutObject"),
                context={},
            )
            await client.meta.events.emit(
                "after-call-error.s3.PutObject",
                exception=_make_client_error_with_request_id("AccessDenied", "req-x"),
                context={},
            )
        assert [call["request_id"] for call in aws_calls] == ["req-emit", "req-x"]


class TestRecordAwsApiCall:
    """The request-scoped accumulator behind the hooks."""

    def test_accumulator_caps_entries_dropping_the_oldest(
        self, aws_calls: list[AwsApiCallLog]
    ) -> None:
        """The accumulator keeps only the newest ``_AWS_REQUESTS_MAX`` entries.

        Ref: stdapi/monitoring.py:record_aws_api_call
        """
        for index in range(_AWS_REQUESTS_MAX + 10):
            record_aws_api_call("s3", "GetObject", f"req-{index}")
        assert len(aws_calls) == _AWS_REQUESTS_MAX
        assert aws_calls[0]["request_id"] == "req-10"
        assert aws_calls[-1]["request_id"] == "req-59"

    def test_record_outside_a_request_scope_is_a_noop(self) -> None:
        """Recording without a request context neither raises nor stores.

        Ref: stdapi/monitoring.py:record_aws_api_call
        """
        assert _AWS_API_CALLS.get(None) is None
        record_aws_api_call("s3", "GetObject", "req-none")
        assert _AWS_API_CALLS.get(None) is None


class TestLogEventDrain:
    """Accumulated entries land on the event that finalizes next."""

    def test_request_log_carries_recorded_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calls recorded in the request scope end up on the "request" event.

        Ref: stdapi/monitoring.py:_attach_aws_api_calls
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        with log_request_event(_make_request()):
            record_aws_api_call("polly", "SynthesizeSpeech", "req-req")
        assert written[-1]["aws_requests"] == [
            {
                "service": "polly",
                "operation": "SynthesizeSpeech",
                "request_id": "req-req",
            }
        ]
        assert _AWS_API_CALLS.get(None) is None

    async def test_stream_log_drains_calls_recorded_mid_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calls recorded while streaming land on the "request_stream" event.

        Ref: stdapi/monitoring.py:_rebuild_and_log_stream
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        id_token = monitoring.REQUEST_ID.set("rid-stream")
        calls_token = _AWS_API_CALLS.set([])
        usage_token = usage.init_usage()
        state_token = usage.init_model_state()
        try:

            async def chunks() -> AsyncGenerator[int]:
                """Record an AWS call between two stream chunks."""
                yield 1
                record_aws_api_call("transcribe", "GetTranscriptionJob", "req-mid")
                yield 2

            stream = await log_request_stream_event(chunks())
            assert [chunk async for chunk in stream] == [1, 2]
        finally:
            usage.MODEL_STATE.reset(state_token)
            usage.USAGE.reset(usage_token)
            _AWS_API_CALLS.reset(calls_token)
            monitoring.REQUEST_ID.reset(id_token)
        assert written[-1]["type"] == "request_stream"
        assert written[-1]["aws_requests"] == [
            {
                "service": "transcribe",
                "operation": "GetTranscriptionJob",
                "request_id": "req-mid",
            }
        ]

    def test_background_log_drains_recorded_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calls recorded by a background task land on the "background" event.

        Ref: stdapi/monitoring.py:log_background_event
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        token = _AWS_API_CALLS.set([])
        try:
            with log_background_event("cleanup", "rid-background"):
                record_aws_api_call("s3", "DeleteObject", "req-background")
        finally:
            _AWS_API_CALLS.reset(token)
        assert written[-1]["aws_requests"] == [
            {
                "service": "s3",
                "operation": "DeleteObject",
                "request_id": "req-background",
            }
        ]


class TestEdgeCorrelationHeaders:
    """Edge correlation headers recorded from the incoming request."""

    def test_edge_headers_are_recorded_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All three edge headers are copied onto the "request" event.

        Ref: stdapi/monitoring.py:log_request_event
        """
        monkeypatch.setattr(monitoring, "write_log_event", lambda _log: None)
        request = _make_request(
            headers=[
                (b"x-amzn-trace-id", b"Root=1-67891233-abcdef012345678912345678"),
                (b"x-amz-apigw-id", b"AbCdEfGhIjKlMnO="),
                (b"x-amz-cf-id", b"ExampleCloudFrontId=="),
            ]
        )
        with log_request_event(request) as log:
            pass
        assert log["amzn_trace_id"] == "Root=1-67891233-abcdef012345678912345678"
        assert log["apigw_request_id"] == "AbCdEfGhIjKlMnO="
        assert log["cloudfront_request_id"] == "ExampleCloudFrontId=="

    def test_absent_edge_headers_add_no_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the headers, none of the edge fields are added to the log.

        Ref: stdapi/monitoring.py:log_request_event
        """
        monkeypatch.setattr(monitoring, "write_log_event", lambda _log: None)
        with log_request_event(_make_request()) as log:
            pass
        assert "amzn_trace_id" not in log
        assert "apigw_request_id" not in log
        assert "cloudfront_request_id" not in log

    def test_edge_header_values_are_sanitized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client-supplied values are stripped of control characters and truncated.

        Ref: stdapi/monitoring.py:_edge_header_value
        """
        monkeypatch.setattr(monitoring, "write_log_event", lambda _log: None)
        request = _make_request(
            headers=[
                (b"x-amzn-trace-id", "B\x01C\x7f".encode("latin-1")),
                (b"x-amz-cf-id", b"A" * 300),
            ]
        )
        with log_request_event(request) as log:
            pass
        assert log["amzn_trace_id"] == "BC"
        assert log["cloudfront_request_id"] == "A" * _EDGE_HEADER_MAX_LENGTH

    def test_edge_headers_are_recorded_over_http(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real HTTP request's edge headers land in its request log.

        Ref: stdapi/monitoring.py:log_request_event
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        response = app_client.get(
            "/v1/edge-header-probe-not-a-route",
            headers={
                "X-Amzn-Trace-Id": "Root=1-abc-123",
                "x-amz-apigw-id": "ApiGwId=",
                "X-Amz-Cf-Id": "CfId==",
            },
        )
        assert response.status_code == 404
        log = written[-1]
        assert log["amzn_trace_id"] == "Root=1-abc-123"
        assert log["apigw_request_id"] == "ApiGwId="
        assert log["cloudfront_request_id"] == "CfId=="
        assert log["id"] not in ("Root=1-abc-123", "ApiGwId=", "CfId==")
