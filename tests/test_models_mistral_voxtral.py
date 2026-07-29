"""Unit tests for Mistral Voxtral: ``include=["logprobs"]`` is accepted but never populated."""

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.audio import mistral_voxtral
from stdapi.models.audio.mistral_voxtral import AudioModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: A Voxtral model ID used across the module's tests.
_MODEL_ID = "mistral.voxtral-mini-3b-2507"


class _FakeAudioContent:
    """Minimal ``InputFile`` stand-in exposing only what ``_build_request`` needs."""

    async def get_content_type_tuple(self) -> tuple[str, str]:
        """Report a fixed audio/mp3 content type."""
        return ("audio", "mp3")

    async def to_base64(self) -> str:
        """Return a fixed base64 payload."""
        return "ZmFrZQ=="


def _fake_response() -> dict[str, Any]:
    """Build a minimal Bedrock Voxtral response payload."""
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "hello world",
                    "refusal": None,
                    "role": "assistant",
                },
            }
        ],
        "created": 0,
        "id": "id",
        "model": _MODEL_ID,
        "object": "chat.completion",
        "service_tier": "standard",
        "usage": {
            "completion_tokens": 2,
            "prompt_tokens": 10,
            "total_tokens": 12,
            "prompt_tokens_details": {"audio_tokens": 10, "cached_tokens": 0},
        },
    }


async def _fake_stream_chunks(
    *_args: object, **_kwargs: object
) -> AsyncGenerator[dict[str, Any]]:
    """Yield a single streaming chunk carrying no log probabilities."""
    yield {
        "choices": [
            {
                "delta": {"content": "hello world"},
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
            }
        ],
        "created": 0,
        "id": "id",
        "model": _MODEL_ID,
        "object": "chat.completion.chunk",
        "service_tier": "standard",
        "amazon-bedrock-invocationMetrics": {
            "inputTokenCount": 10,
            "outputTokenCount": 2,
            "invocationLatency": 1,
            "firstByteLatency": 1,
        },
    }


class TestSttAcceptsLogprobs:
    """stt(): ``include=["logprobs"]`` is accepted and always returns ``logprobs=None``."""

    @pytest.mark.parametrize("logprobs", [True, False])
    async def test_logprobs_is_always_none(
        self, monkeypatch: pytest.MonkeyPatch, logprobs: bool
    ) -> None:
        """Requesting logprobs reaches Bedrock and the response reports logprobs=None."""

        async def _fake_invoke(self: AudioModel, _request: object) -> InvokeResult[Any]:  # noqa: ARG001
            return InvokeResult(response=_fake_response())

        monkeypatch.setattr(mistral_voxtral.AudioModel, "invoke", _fake_invoke)

        model = AudioModel(_MODEL_ID)
        response = await model.stt(_FakeAudioContent(), "json", logprobs=logprobs)  # type: ignore[arg-type]

        assert response.text == "hello world"  # type: ignore[union-attr]
        assert response.logprobs is None  # type: ignore[union-attr]


class TestSttStreamAcceptsLogprobs:
    """stt_stream(): ``include=["logprobs"]`` is accepted and always returns ``logprobs=None``."""

    @pytest.mark.parametrize("logprobs", [True, False])
    async def test_logprobs_is_always_none(
        self, monkeypatch: pytest.MonkeyPatch, logprobs: bool
    ) -> None:
        """Requesting logprobs reaches Bedrock and every event reports logprobs=None."""
        monkeypatch.setattr(
            mistral_voxtral.AudioModel, "invoke_stream", _fake_stream_chunks
        )

        model = AudioModel(_MODEL_ID)
        events = [
            event
            async for event in model.stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                logprobs=logprobs,
            )
        ]

        assert all(event.logprobs is None for event in events)
        assert events[0].delta == "hello world"  # type: ignore[union-attr]
