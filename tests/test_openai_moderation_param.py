"""Tests for the request-level moderation parameter on chat and responses."""

from typing import Any

import pytest
from starlette.testclient import TestClient

from stdapi.aws_bedrock import GUARDRAIL_TRACE_VAR, GUARDTRAIL_CONFIG_VAR
from stdapi.config import SETTINGS
from stdapi.models import ModelDetails
from stdapi.routes import openai_chat_completions, openai_responses
from stdapi.types.openai_chat_completions import ChatCompletion
from stdapi.types.openai_responses import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseUsage,
)

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
        self.guardrail_configs.append(GUARDTRAIL_CONFIG_VAR.get(None))
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
    ) -> Response:
        """Capture context and return a canned response."""
        self._capture()
        return Response(
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


@pytest.fixture
def chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatBackend:
    """Stub model validation and both generation backends."""
    stub = _StubChatBackend()
    for module in (openai_chat_completions, openai_responses):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_chat_model", lambda _model_id: stub)
    return stub


def _assert_moderation(moderation: dict[str, Any]) -> None:
    assert moderation["input"]["flagged"] is True
    assert moderation["input"]["categories"] == {"sexual": True}
    assert moderation["input"]["category_scores"] == {"sexual": 0.75}
    assert moderation["input"]["model"] == "gr123"
    assert moderation["output"]["flagged"] is False
    assert moderation["output"]["categories"] == {"violence": False}
    assert moderation["output"]["category_scores"] == {"violence": 0.25}


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
        _assert_moderation(response.json()["moderation"])

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
        assert moderation["input"]["flagged"] is True

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
