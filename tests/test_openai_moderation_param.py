"""Tests for the request-level moderation parameter on chat and responses.

Unlike /v1/moderations, this gateway extension derives its results from the
Converse guardrail trace, so only the categories the trace actually reports are
present (not the full 13-key OpenAI vocabulary).

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailTraceAssessment.html
     stdapi/routes/_moderation.py:apply_request_moderation
"""

from json import loads
from typing import TYPE_CHECKING, Any, cast

import pytest

import stdapi.models
from stdapi.aws_bedrock import GUARDRAIL_CONFIG_VAR, GUARDRAIL_TRACE_VAR
from stdapi.config import SETTINGS
from stdapi.models import ModelBase
from stdapi.models.chat._adapters import _openai_chat_completion as chat_adapter
from stdapi.monitoring import REQUEST_LOG
from stdapi.routes import openai_chat_completions, openai_responses
from stdapi.routes._moderation import apply_request_moderation
from stdapi.types.openai import RequestModeration
from stdapi.types.openai_chat_completions import ChatCompletion
from stdapi.types.openai_responses import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseUsage,
)
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from starlette.testclient import TestClient
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


#: Guardrail trace: flagged input (object) and clean output (list), as AWS shapes them.
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
    return make_model_details(model_id)


@pytest.fixture
def configured_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a default server guardrail, as ``moderation: true`` requires."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")


@pytest.mark.usefixtures("configured_guardrail")
class TestApplyRequestModerationTraceMode:
    """apply_request_moderation requests the full guardrail trace.

    ``trace: enabled`` only reports assessments for flagged categories, which
    zeroes out every non-detected category's score once mapped by
    ``map_guardrail_filters``; ``enabled_full`` is required to report real
    confidence for every category, matching the ``/v1/moderations`` fix.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/routes/_moderation.py:apply_request_moderation
    """

    def test_guardrail_config_requests_full_trace(self) -> None:
        """The guardrail config set on the request carries trace=enabled_full."""
        config_token = GUARDRAIL_CONFIG_VAR.set({})
        trace_token = GUARDRAIL_TRACE_VAR.set({})
        try:
            apply_request_moderation(RequestModeration(model="gr123"))
            config = GUARDRAIL_CONFIG_VAR.get(None)
        finally:
            GUARDRAIL_CONFIG_VAR.reset(config_token)
            GUARDRAIL_TRACE_VAR.reset(trace_token)
        assert config is not None
        assert config["trace"] == "enabled_full"


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


@pytest.mark.usefixtures("configured_guardrail")
class TestChatModerationParam:
    """moderation parameter on POST /v1/chat/completions.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/routes/_moderation.py:build_chat_moderation
    """

    def test_moderation_sets_guardrail_and_reports_results(
        self, app_client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """The guardrail config is applied and trace results are reported.

        The Converse request must carry ``trace: enabled_full``, otherwise AWS
        only reports assessments for flagged categories and every non-flagged
        category score comes back zeroed. Both directions are reported as
        ``moderation_results`` sets even though Converse spells the input
        assessment as an object and the output ones as a list.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
             stdapi/routes/_moderation.py:_trace_results
        """
        response = app_client.post(
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
            "trace": "enabled_full",
        }
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "hi", (
            "moderation must not alter the completion"
        )
        moderation = body["moderation"]
        for direction in ("input", "output"):
            assert moderation[direction]["type"] == "moderation_results"
            assert moderation[direction]["model"] == "gr123"
        (input_result,) = moderation["input"]["results"]
        _assert_input_result(input_result)
        (output_result,) = moderation["output"]["results"]
        _assert_output_result(output_result)

    def test_without_moderation_no_field(
        self, app_client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """Without the parameter no moderation field is reported.

        The field is opt-in: an otherwise identical completion comes back
        complete but with no ``moderation`` key at all, rather than an empty or
        null one.

        Ref: stdapi/routes/_moderation.py:build_chat_moderation
        """
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "moderation" not in body
        # The completion itself is still fully reported.
        assert body["choices"][0]["message"]["content"] == "hi"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_comprehend_model_is_rejected(
        self, app_client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """The Comprehend moderation model is not usable as request parameter.

        Request-level moderation rides on the Converse guardrail trace, which
        Comprehend cannot produce, so the model is refused before any generation
        happens rather than silently ignored.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/routes/_moderation.py:apply_request_moderation
        """
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "amazon.comprehend-toxicity"},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "guardrail" in error["message"]
        assert "Moderations API" in error["message"]
        assert not chat_backend.guardrail_configs, "no generation must have happened"

    def test_default_guardrail_model_id(
        self, app_client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """amazon.bedrock-runtime-guardrail selects the configured guardrail.

        The synthetic model ID resolves to the server's guardrail while the
        reported ``model`` echoes what the client asked for.

        Ref: stdapi/aws_bedrock.py:GUARDRAIL_MODERATION_MODEL
             stdapi/aws_bedrock.py:resolve_guardrail_model
        """
        response = app_client.post(
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
        assert config["guardrailVersion"] == "1"
        assert config["trace"] == "enabled_full"
        moderation = response.json()["moderation"]
        assert moderation["input"]["model"] == "amazon.bedrock-runtime-guardrail"
        assert moderation["output"]["model"] == "amazon.bedrock-runtime-guardrail"

    def test_text_moderation_model_is_rejected(
        self, app_client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """text-moderation-* aliases Comprehend and is rejected here.

        The OpenAI legacy alias inherits the Comprehend restriction, so it fails
        with the same guardrail-required error as the explicit model ID.

        Ref: stdapi/aws_bedrock.py:is_comprehend_moderation_model
             stdapi/routes/_moderation.py:apply_request_moderation
        """
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "text-moderation-latest"},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "guardrail" in error["message"]
        assert "Moderations API" in error["message"]
        assert not chat_backend.guardrail_configs

    def test_no_guardrail_configured_hides_settings(
        self,
        app_client: TestClient,
        chat_backend: _StubChatBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a configured guardrail the 400 does not expose settings.

        Error messages are the one place server configuration could leak to
        clients, so the setting name is replaced by a pointer to the
        administrator. There is no Comprehend fallback here, unlike /moderations.

        Ref: stdapi/aws_bedrock.py:resolve_guardrail_model
             stdapi/utils.py:hide_security_details
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "omni-moderation-latest"},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        message = error["message"]
        assert "administrator" in message
        assert "guardrail" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()
        assert not chat_backend.guardrail_configs


@pytest.mark.usefixtures("configured_guardrail")
class TestResponsesModerationParam:
    """moderation parameter on POST /v1/responses.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/routes/_moderation.py:build_response_moderation
    """

    def test_moderation_sets_guardrail_and_reports_results(
        self, app_client: TestClient, chat_backend: _StubChatBackend
    ) -> None:
        """The guardrail config is applied and trace results are reported.

        The Responses shape reports one ``moderation_result`` per direction
        directly, where Chat Completions nests a ``results`` list; both are fed
        by the same guardrail trace.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailTraceAssessment.html
             stdapi/routes/_moderation.py:_to_moderation_result
        """
        response = app_client.post(
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
            "trace": "enabled_full",
        }
        moderation = response.json()["moderation"]
        assert moderation["input"]["model"] == "omni-moderation-latest"
        assert moderation["input"]["type"] == "moderation_result"
        assert moderation["input"]["flagged"] is True
        assert moderation["input"]["categories"] == {"sexual": True}
        assert moderation["input"]["category_scores"] == {"sexual": 0.75}
        assert moderation["input"]["category_applied_input_types"] == {
            "sexual": ["text"]
        }
        assert moderation["output"]["model"] == "omni-moderation-latest"
        assert moderation["output"]["type"] == "moderation_result"
        assert moderation["output"]["flagged"] is False
        # A LOW, non-blocking output filter: scored but not flagged.
        assert moderation["output"]["categories"] == {"violence": False}
        assert moderation["output"]["category_scores"] == {"violence": 0.25}
        assert moderation["output"]["category_applied_input_types"] == {
            "violence": ["text"]
        }

    def test_unknown_guardrail_override_rejected(
        self,
        app_client: TestClient,
        chat_backend: _StubChatBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit guardrail is rejected when overrides are not allowed.

        Client-chosen guardrails are refused unless the operator opted in, and the
        request is stopped before generation so no tokens are billed.

        Ref: stdapi/aws_bedrock.py:resolve_guardrail_model
             stdapi/config.py:_Settings.aws_bedrock_allow_guardrail_override
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "moderation": {"model": "other456:2"},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "not allowed" in error["message"]
        assert not chat_backend.guardrail_configs

    def test_no_guardrail_configured_hides_settings(
        self,
        app_client: TestClient,
        chat_backend: _StubChatBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a configured guardrail the 400 does not expose settings.

        Same operator-privacy rule as Chat Completions: the message names the
        administrator, never ``aws_bedrock_guardrail_identifier``.

        Ref: stdapi/aws_bedrock.py:resolve_guardrail_model
             stdapi/utils.py:hide_security_details
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "moderation": {"model": "omni-moderation-latest"},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        message = error["message"]
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
    """converse()/converse_stream() capture trace.guardrail into the shared holder.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
         stdapi/aws_bedrock.py:GUARDRAIL_TRACE_VAR
    """

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
        """The non-streaming wrapper merges the response trace into the holder.

        The holder is a per-request ContextVar because the trace is produced deep
        inside the model call but consumed by the route when it builds the
        response's ``moderation`` field.

        Ref: stdapi/models/__init__.py:ModelBase.converse
        """
        holder: dict[str, Any] = {}
        token = GUARDRAIL_TRACE_VAR.set(holder)
        try:
            response = await ModelBase("tracemodel").converse({"modelId": "tracemodel"})
        finally:
            GUARDRAIL_TRACE_VAR.reset(token)
        assert holder == _TRACE
        assert response["stopReason"] == "end_turn", "the response is still returned"

    async def test_converse_stream_updates_the_trace_holder(self) -> None:
        """Consuming the stream merges the metadata event trace into the holder.

        ConverseStream only reports the guardrail trace in its trailing metadata
        event, so the holder stays empty until the stream is fully consumed.

        Ref: stdapi/models/__init__.py:ModelBase.converse_stream
        """
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
        """Without an installed holder the trace is simply not captured.

        Requests that did not ask for moderation must not pay any bookkeeping
        cost, and the raw AWS trace still reaches the caller untouched.

        Ref: stdapi/models/__init__.py:ModelBase.converse
        """
        response = await ModelBase("tracemodel").converse({"modelId": "tracemodel"})
        assert response["trace"]["guardrail"] == _TRACE
        assert response["stopReason"] == "end_turn"
        assert GUARDRAIL_TRACE_VAR.get(None) is None


class TestChatStreamingModerationDrop:
    """Streaming chat completions carry no moderation payload (documented drop).

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/routes/_moderation.py:build_chat_moderation
    """

    async def test_no_moderation_on_any_chunk(self) -> None:
        """Even with a captured trace, no streamed chunk carries moderation.

        The guardrail trace only arrives in ConverseStream's trailing metadata
        event, by which time earlier chunks are already sent, so the gateway
        deliberately omits ``moderation`` from every chunk rather than emitting it
        late on the last one.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
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
        payloads: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk == "[DONE]":
                continue
            payload = loads(chunk) if isinstance(chunk, str) else chunk
            assert isinstance(payload, dict)
            assert "moderation" not in payload  # ...but never reported on chunks.
            payloads.append(payload)
        assert payloads
        # The stream itself is intact: the text delta and the finish reason are there.
        choices = [
            choice for payload in payloads for choice in payload.get("choices", [])
        ]
        assert any(choice.get("delta", {}).get("content") == "hi" for choice in choices)
        assert any(choice.get("finish_reason") == "stop" for choice in choices)
