"""Unit tests for the Mistral Voxtral speech-to-text adapter.

Bedrock and the response are stubbed, so the token-usage mapping and the
``logprobs`` handling are exercised without any AWS call.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
     stdapi/models/audio/mistral_voxtral.py:AudioModel
"""

from typing import TYPE_CHECKING, Any

import pytest
from starlette.responses import Response

from stdapi.models import InvokeResult
from stdapi.models.audio import mistral_voxtral
from stdapi.models.audio.mistral_voxtral import AudioModel
from stdapi.types.openai_audio import UsageTokens

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
            # cached_tokens is deliberately far from prompt_tokens - audio_tokens
            # (10 - 7 = 3) so a regression back to reading cached_tokens as
            # text_tokens is caught (issue #95).
            "prompt_tokens_details": {"audio_tokens": 7, "cached_tokens": 5},
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
    """Non-streaming ``stt()``: transcript, token usage and ``logprobs`` mapping.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
    """

    @pytest.mark.parametrize("logprobs", [True, False])
    async def test_logprobs_is_always_none(
        self, monkeypatch: pytest.MonkeyPatch, logprobs: bool
    ) -> None:
        """``logprobs`` stays ``None`` whether or not it was requested.

        Bedrock reports ``logprobs: null`` on every choice, so requesting them cannot
        populate the field. The same response also pins the usage mapping: Bedrock's
        ``prompt_tokens``/``completion_tokens`` become OpenAI's
        ``input_tokens``/``output_tokens``, and ``input_token_details.text_tokens``
        is derived as ``prompt_tokens - audio_tokens`` rather than read from
        ``prompt_tokens_details.cached_tokens`` (issue #95).

        Ref: stdapi/types/openai_audio.py:UsageTokens
             https://github.com/stdapi-ai/stdapi.ai/issues/95
        """

        async def _fake_invoke(self: AudioModel, _request: object) -> InvokeResult[Any]:  # noqa: ARG001
            return InvokeResult(response=_fake_response())

        monkeypatch.setattr(mistral_voxtral.AudioModel, "invoke", _fake_invoke)

        model = AudioModel(_MODEL_ID)
        response = await model.stt(_FakeAudioContent(), "json", logprobs=logprobs)  # type: ignore[arg-type]

        assert response.text == "hello world"  # type: ignore[union-attr]
        assert response.logprobs is None  # type: ignore[union-attr]

        usage = response.usage  # type: ignore[union-attr]
        assert isinstance(usage, UsageTokens)
        assert usage.type == "tokens"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 2
        assert usage.total_tokens == 12
        assert usage.input_token_details is not None
        assert usage.input_token_details.audio_tokens == 7
        assert usage.input_token_details.text_tokens == 3

    async def test_input_token_details_omitted_when_audio_tokens_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``input_token_details`` is omitted when Bedrock reports ``audio_tokens=0``.

        A captured real Voxtral response reported ``prompt_tokens=384`` with
        ``prompt_tokens_details.audio_tokens=0`` for an audio-dominated request, which
        would make ``text_tokens = prompt_tokens - audio_tokens`` attribute the whole
        prompt to text. Omitting the breakdown avoids reporting that wrong split
        (issue #95).

        Ref: stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
             https://github.com/stdapi-ai/stdapi.ai/issues/95
        """
        response = _fake_response()
        response["usage"]["prompt_tokens_details"] = {
            "audio_tokens": 0,
            "cached_tokens": 5,
        }

        async def _fake_invoke(self: AudioModel, _request: object) -> InvokeResult[Any]:  # noqa: ARG001
            return InvokeResult(response=response)

        monkeypatch.setattr(mistral_voxtral.AudioModel, "invoke", _fake_invoke)

        model = AudioModel(_MODEL_ID)
        result = await model.stt(_FakeAudioContent(), "json", logprobs=False)  # type: ignore[arg-type]

        usage = result.usage  # type: ignore[union-attr]
        assert isinstance(usage, UsageTokens)
        assert usage.input_tokens == 10
        assert usage.input_token_details is None

    async def test_input_token_details_omitted_when_audio_tokens_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response with ``prompt_tokens_details`` but no ``audio_tokens`` key does not crash.

        ``_PromptTokensDetails`` is ``total=False``, so a real Bedrock response can
        omit ``audio_tokens`` entirely rather than reporting it as ``0``. Indexing
        the key directly would raise ``KeyError`` and turn a successful
        transcription into a 500.

        Ref: stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
        """
        response = _fake_response()
        response["usage"]["prompt_tokens_details"] = {"cached_tokens": 5}

        async def _fake_invoke(self: AudioModel, _request: object) -> InvokeResult[Any]:  # noqa: ARG001
            return InvokeResult(response=response)

        monkeypatch.setattr(mistral_voxtral.AudioModel, "invoke", _fake_invoke)

        model = AudioModel(_MODEL_ID)
        result = await model.stt(_FakeAudioContent(), "json", logprobs=False)  # type: ignore[arg-type]

        usage = result.usage  # type: ignore[union-attr]
        assert isinstance(usage, UsageTokens)
        assert usage.input_tokens == 10
        assert usage.input_token_details is None


class TestSttTextFormat:
    """``stt()``/``stt_translate()`` with ``response_format=text``: raw ``text/plain``.

    A bare ``Response`` must be returned for this format so FastAPI does not
    JSON-encode the transcript as a quoted string; a prior regression
    (``return content``) still passed the existing word-match assertion because
    FastAPI's default JSON encoding of a bare ``str`` also contains the words,
    quotes and all.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
    """

    async def test_transcription_text_format_is_a_raw_plain_text_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``stt()`` returns a ``Response`` with the exact, unquoted transcript body."""

        async def _fake_invoke(self: AudioModel, _request: object) -> InvokeResult[Any]:  # noqa: ARG001
            return InvokeResult(response=_fake_response())

        monkeypatch.setattr(mistral_voxtral.AudioModel, "invoke", _fake_invoke)

        model = AudioModel(_MODEL_ID)
        response = await model.stt(_FakeAudioContent(), "text", logprobs=False)  # type: ignore[arg-type]

        assert isinstance(response, Response)
        assert response.media_type == "text/plain; charset=utf-8"
        assert response.body == b"hello world"


class TestSttStreamAcceptsLogprobs:
    """Streaming ``stt_stream()``: event sequence, usage from Bedrock metrics, logprobs.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/mistral_voxtral.py:AudioModel.stt_stream
    """

    @pytest.mark.parametrize("logprobs", [True, False])
    async def test_logprobs_is_always_none(
        self, monkeypatch: pytest.MonkeyPatch, logprobs: bool
    ) -> None:
        """Every streamed event reports ``logprobs=None``, requested or not.

        The single stubbed chunk yields one ``transcript.text.delta`` followed by the
        terminal ``transcript.text.done``, whose usage is built from
        ``amazon-bedrock-invocationMetrics``. That footer carries no audio/text
        split, so ``input_token_details`` stays unset rather than attributing every
        input token to audio as before (issue #95).

        Ref: stdapi/types/openai_audio.py:TranscriptionTextDoneEvent
             https://github.com/stdapi-ai/stdapi.ai/issues/95
        """
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
        assert [event.type for event in events] == [
            "transcript.text.delta",
            "transcript.text.done",
        ]
        assert events[0].delta == "hello world"  # type: ignore[union-attr]

        done = events[-1]
        assert done.text == "hello world"  # type: ignore[union-attr]
        usage = done.usage  # type: ignore[union-attr]
        assert usage is not None
        assert usage.type == "tokens"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 2
        assert usage.total_tokens == 12
        assert usage.input_token_details is None
