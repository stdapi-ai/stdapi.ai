"""Tests for the request-level moderation parameter on chat and responses."""

from json import loads
from typing import TYPE_CHECKING, Any, cast

import pytest
from starlette.testclient import TestClient

import stdapi.models
from stdapi.aws_bedrock import GUARDRAIL_CONFIG_VAR, GUARDRAIL_TRACE_VAR
from stdapi.config import SETTINGS
from stdapi.models import ModelBase, ModelDetails
from stdapi.models.chat._adapters import _openai_chat_completion as chat_adapter
from stdapi.monitoring import REQUEST_LOG
from stdapi.routes import openai_chat_completions, openai_responses
from stdapi.types.openai_chat_completions import ChatCompletion
from stdapi.types.openai_responses import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


#: Guardrail trace with a flagged input and a clean output assessment.
_TRACE: dict[str, Any] = {
    "inputAssessment": {
        "gr123": {
            "contentPolicy": {
                "filters": [
                    {"type": "SEXUAL", "confidence": "HIGH", "action": "BLOCKED"}
                ]
            }
        }
    },
    "outputAssessments": {
        "gr123": [
            {
                "contentPolicy": {
                    "filters": [
                        {"type": "VIOLENCE", "confidence": "LOW", "action": "NONE"}
                    ]
                }
            }
        ]
    },
}


async def _validate_model(model_id: str, *_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
    return ModelDetails(
        id=model_id,
        name=model_id,
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")
    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


class _StubChatBackend:
    """Stub chat backend capturing the guardrail config and setting a trace."""

    def __init__(self) -> None:
        self.guardrail_configs: list[Any] = []

    def _capture(self) -> None:
        self.guardrail_configs.append(GUARDRAIL_CONFIG_VAR.get(None))
        if (holder := GUARDRAIL_TRACE_VAR.get(None)) is not None:
            holder.update(_TRACE)

    def native_store_supported(self) -> bool:
        """Local-store stub: no Mantle native storage."""
        return False

    async def create_completion(
        self,
        request: Any,  # noqa: ANN401
        completion_id: str,
        created: int,
    ) -> ChatCompletion:
        """Capture context and return a canned completion."""
        self._capture()
        return ChatCompletion.model_validate(
            {
                "id": completion_id,
                "created": created,
                "model": request.model,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def create_response(
        self,
        request: Any,  # noqa: ANN401
        response_id: str,
        created_at: float,
        moderation_builder: Any = None,  # noqa: ANN401
    ) -> Response:
        """Capture context and return a canned response, honoring the builder contract."""
        self._capture()
        response = Response(
            id=response_id,
            created_at=created_at,
            model=request.model,
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            usage=ResponseUsage(
                input_tokens=1,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=1,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=2,
            ),
        )
        if moderation_builder is not None:
            response.moderation = moderation_builder()
        return response


@pytest.fixture
def chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatBackend:
    """Stub model validation and both generation backends."""
    stub = _StubChatBackend()
    for module in (openai_chat_completions, openai_responses):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_chat_model", lambda _model_id: stub)
    return stub


def _assert_input_result(result: dict[str, Any]) -> None:
    assert result["type"] == "moderation_result"
    assert result["flagged"] is True
    assert result["categories"] == {"sexual": True}
    assert result["category_scores"] == {"sexual": 0.75}
    assert result["category_applied_input_types"] == {"sexual": ["text"]}
    assert result["model"] == "gr123"


def _assert_output_result(result: dict[str, Any]) -> None:
    assert result["type"] == "moderation_result"
    assert result["flagged"] is False
    assert result["categories"] == {"violence": False}
    assert result["category_scores"] == {"violence": 0.25}
    assert result["category_applied_input_types"] == {"violence": ["text"]}
    assert result["model"] == "gr123"


class TestChatModerationParam:
    """moderation parameter on POST /v1/chat/completions."""

    def test_moderation_sets_guardrail_and_reports_results(
        self, client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """The guardrail config is applied and trace results are reported."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "gr123"},
            },
        )
        assert response.status_code == 200, response.text
        (config,) = chat_backend.guardrail_configs
        assert config == {
            "guardrailIdentifier": "gr123",
            "guardrailVersion": "1",
            "trace": "enabled",
        }
        moderation = response.json()["moderation"]
        for direction in ("input", "output"):
            assert moderation[direction]["type"] == "moderation_results"
            assert moderation[direction]["model"] == "gr123"
        (input_result,) = moderation["input"]["results"]
        _assert_input_result(input_result)
        (output_result,) = moderation["output"]["results"]
        _assert_output_result(output_result)

    def test_without_moderation_no_field(
        self, client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """Without the parameter no moderation field is reported."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        assert "moderation" not in response.json()

    def test_comprehend_model_is_rejected(
        self, client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """The Comprehend moderation model is not usable as request parameter."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "amazon.comprehend-toxicity"},
            },
        )
        assert response.status_code == 400
        assert "guardrail" in response.json()["error"]["message"]
        assert not chat_backend.guardrail_configs

    def test_default_guardrail_model_id(
        self, client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """amazon.bedrock-runtime-guardrail selects the configured guardrail."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "amazon.bedrock-runtime-guardrail"},
            },
        )
        assert response.status_code == 200, response.text
        (config,) = chat_backend.guardrail_configs
        assert config["guardrailIdentifier"] == "gr123"
        moderation = response.json()["moderation"]
        assert moderation["input"]["model"] == "amazon.bedrock-runtime-guardrail"

    def test_text_moderation_model_is_rejected(
        self, client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """text-moderation-* aliases Comprehend and is rejected here."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "text-moderation-latest"},
            },
        )
        assert response.status_code == 400
        assert "guardrail" in response.json()["error"]["message"]
        assert not chat_backend.guardrail_configs

    def test_no_guardrail_configured_hides_settings(
        self,
        client: TestClient,
        chat_backend: _StubChatBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a configured guardrail the 400 does not expose settings."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "omni-moderation-latest"},
            },
        )
        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "administrator" in message
        assert "guardrail" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()
        assert not chat_backend.guardrail_configs


class TestResponsesModerationParam:
    """moderation parameter on POST /v1/responses."""

    def test_moderation_sets_guardrail_and_reports_results(
        self, client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """The guardrail config is applied and trace results are reported."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "moderation": {"model": "omni-moderation-latest"},
            },
        )
        assert response.status_code == 200, response.text
        (config,) = chat_backend.guardrail_configs
        assert config == {
            "guardrailIdentifier": "gr123",
            "guardrailVersion": "1",
            "trace": "enabled",
        }
        moderation = response.json()["moderation"]
        assert moderation["input"]["model"] == "omni-moderation-latest"
        assert moderation["input"]["type"] == "moderation_result"
        assert moderation["input"]["flagged"] is True
        assert moderation["input"]["category_applied_input_types"] == {
            "sexual": ["text"]
        }
        assert moderation["output"]["flagged"] is False
        assert moderation["output"]["category_applied_input_types"] == {
            "violence": ["text"]
        }

    def test_unknown_guardrail_override_rejected(
        self,
        client: TestClient,
        chat_backend: _StubChatBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit guardrail is rejected when overrides are not allowed."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "moderation": {"model": "other456:2"},
            },
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["error"]["message"]
        assert not chat_backend.guardrail_configs

    def test_no_guardrail_configured_hides_settings(
        self,
        client: TestClient,
        chat_backend: _StubChatBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a configured guardrail the 400 does not expose settings."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "moderation": {"model": "omni-moderation-latest"},
            },
        )
        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "administrator" in message
        assert "guardrail" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()
        assert not chat_backend.guardrail_configs


async def _noop_prepare(_self: object, _request: object, _region: str) -> None:
    """No-op stand-in for ModelBase._prepare_converse_request_for_region."""


async def _events(events: list[dict[str, Any]]) -> AsyncGenerator[dict[str, Any]]:
    """Yield fabricated Bedrock ConverseStream events."""
    for event in events:
        yield event


class _StubTraceClient:
    """Fake Bedrock client whose responses carry a guardrail trace."""

    async def converse(self, **_kwargs: object) -> dict[str, Any]:
        """Return a canned Converse response with trace.guardrail."""
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "trace": {"guardrail": _TRACE},
        }

    async def converse_stream(self, **_kwargs: object) -> dict[str, Any]:
        """Return a canned ConverseStream response with a trailing trace."""
        return {
            "stream": _events(
                [
                    {"contentBlockDelta": {"delta": {"text": "hi"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {
                        "metadata": {
                            "usage": {
                                "inputTokens": 1,
                                "outputTokens": 1,
                                "totalTokens": 2,
                            },
                            "trace": {"guardrail": _TRACE},
                        }
                    },
                ]
            )
        }


class TestGuardrailTraceCapture:
    """converse()/converse_stream() capture trace.guardrail into the shared holder."""

    @pytest.fixture(autouse=True)
    def _stub_bedrock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route the real converse wrappers to the stub client in one region."""

        async def _candidates(_model_id: str, **_kwargs: object) -> list[str]:
            return ["us-east-1"]

        monkeypatch.setattr(stdapi.models, "compute_candidate_regions", _candidates)
        monkeypatch.setattr(
            ModelBase, "_prepare_converse_request_for_region", _noop_prepare
        )
        monkeypatch.setattr(
            stdapi.models,
            "bedrock_client",
            lambda _region, **_kwargs: _StubTraceClient(),
        )

    async def test_converse_updates_the_trace_holder(self) -> None:
        """The non-streaming wrapper merges the response trace into the holder."""
        holder: dict[str, Any] = {}
        token = GUARDRAIL_TRACE_VAR.set(holder)
        try:
            await ModelBase("tracemodel").converse({"modelId": "tracemodel"})
        finally:
            GUARDRAIL_TRACE_VAR.reset(token)
        assert holder == _TRACE

    async def test_converse_stream_updates_the_trace_holder(self) -> None:
        """Consuming the stream merges the metadata event trace into the holder."""
        holder: dict[str, Any] = {}
        token = GUARDRAIL_TRACE_VAR.set(holder)
        try:
            response = await ModelBase("tracemodel").converse_stream(
                {"modelId": "tracemodel"}
            )
            assert holder == {}  # Nothing captured before the metadata event.
            async for _ in response["stream"]:
                pass
        finally:
            GUARDRAIL_TRACE_VAR.reset(token)
        assert holder == _TRACE

    async def test_converse_without_holder_is_a_no_op(self) -> None:
        """Without an installed holder the trace is simply not captured."""
        response = await ModelBase("tracemodel").converse({"modelId": "tracemodel"})
        assert response["trace"]["guardrail"] == _TRACE
        assert GUARDRAIL_TRACE_VAR.get(None) is None


class TestChatStreamingModerationDrop:
    """Streaming chat completions carry no moderation payload (documented drop)."""

    async def test_no_moderation_on_any_chunk(self) -> None:
        """Even with a captured trace, no streamed chunk carries moderation."""
        events: list[dict[str, Any]] = [
            {"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "trace": {"guardrail": _TRACE},
                }
            },
        ]
        holder: dict[str, Any] = {}
        trace_token = GUARDRAIL_TRACE_VAR.set(holder)
        legacy_token = chat_adapter._LEGACY_FUNCTION.set(False)  # noqa: SLF001
        log_token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
        try:
            stream = ModelBase("tracemodel")._capture_stream_usage(  # noqa: SLF001
                cast("AsyncGenerator[ConverseStreamOutputTypeDef]", _events(events))
            )
            chunks = [
                sse.data
                async for sse in chat_adapter.format_stream(
                    "chatcmpl-1", 1, "tracemodel", stream, None
                )
            ]
        finally:
            REQUEST_LOG.reset(log_token)
            chat_adapter._LEGACY_FUNCTION.reset(legacy_token)  # noqa: SLF001
            GUARDRAIL_TRACE_VAR.reset(trace_token)
        assert holder == _TRACE  # The trace was captured...
        assert chunks
        for chunk in chunks:
            if chunk == "[DONE]":
                continue
            payload = loads(chunk) if isinstance(chunk, str) else chunk
            assert isinstance(payload, dict)
            assert "moderation" not in payload  # ...but never reported on chunks.
