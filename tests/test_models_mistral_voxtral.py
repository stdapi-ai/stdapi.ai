"""Unit tests for the Mistral Voxtral speech-to-text adapter.

Bedrock Converse and its response are stubbed, so the token-usage mapping and
the ``logprobs`` handling are exercised without any AWS call.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
     stdapi/models/audio/_default.py:AudioModel
"""

from typing import TYPE_CHECKING, Any

import pytest
from starlette.responses import Response

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

    async def to_bytes(self) -> bytes:
        """Return a fixed audio payload."""
        return b"fake"


def _fake_converse_response() -> dict[str, Any]:
    """Build a minimal Bedrock Converse response payload."""
    return {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "hello world"}]}
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
    }


async def _fake_stream_events() -> AsyncGenerator[dict[str, Any]]:
    """Yield the ConverseStream events of a single-delta transcription."""
    yield {"messageStart": {"role": "assistant"}}
    yield {
        "contentBlockDelta": {"delta": {"text": "hello world"}, "contentBlockIndex": 0}
    }
    yield {"contentBlockStop": {"contentBlockIndex": 0}}
    yield {"messageStop": {"stopReason": "end_turn"}}
    yield {
        "metadata": {"usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}}
    }


async def _fake_converse_stream(*_args: object, **_kwargs: object) -> dict[str, Any]:
    """Return a stubbed ConverseStream response wrapping the fake event stream."""
    return {"stream": _fake_stream_events()}


class TestSttAcceptsLogprobs:
    """Non-streaming ``stt()``: transcript, token usage and ``logprobs`` mapping.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/_default.py:AudioModel.stt
    """

    @pytest.mark.parametrize("logprobs", [True, False])
    async def test_logprobs_is_always_none(
        self, monkeypatch: pytest.MonkeyPatch, logprobs: bool
    ) -> None:
        """``logprobs`` stays ``None`` whether or not it was requested.

        Bedrock reports no log probabilities on the Converse API, so requesting
        them cannot populate the field. The same response also pins the usage
        mapping: Converse ``inputTokens``/``outputTokens`` become OpenAI's
        ``input_tokens``/``output_tokens``, and ``input_token_details`` stays
        unset because Converse usage has no audio/text input split (issue #95).

        Ref: stdapi/types/openai_audio.py:UsageTokens
             https://github.com/stdapi-ai/stdapi.ai/issues/95
        """

        async def _fake_converse(self: AudioModel, _request: object) -> dict[str, Any]:  # noqa: ARG001
            return _fake_converse_response()

        monkeypatch.setattr(mistral_voxtral.AudioModel, "converse", _fake_converse)

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
        assert usage.input_token_details is None

    async def test_usage_omitted_when_converse_reports_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Converse response without a ``usage`` block yields ``usage=None``.

        Ref: stdapi/models/audio/_default.py:AudioModel.stt
        """
        response = _fake_converse_response()
        del response["usage"]

        async def _fake_converse(self: AudioModel, _request: object) -> dict[str, Any]:  # noqa: ARG001
            return response

        monkeypatch.setattr(mistral_voxtral.AudioModel, "converse", _fake_converse)

        model = AudioModel(_MODEL_ID)
        result = await model.stt(_FakeAudioContent(), "json", logprobs=False)  # type: ignore[arg-type]

        assert result.text == "hello world"  # type: ignore[union-attr]
        assert result.usage is None  # type: ignore[union-attr]


class TestSttTextFormat:
    """``stt()``/``stt_translate()`` with ``response_format=text``: raw ``text/plain``.

    A bare ``Response`` must be returned for this format so FastAPI does not
    JSON-encode the transcript as a quoted string; a prior regression
    (``return content``) still passed the existing word-match assertion because
    FastAPI's default JSON encoding of a bare ``str`` also contains the words,
    quotes and all.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/_default.py:AudioModel.stt
    """

    async def test_transcription_text_format_is_a_raw_plain_text_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``stt()`` returns a ``Response`` with the exact, unquoted transcript body."""

        async def _fake_converse(self: AudioModel, _request: object) -> dict[str, Any]:  # noqa: ARG001
            return _fake_converse_response()

        monkeypatch.setattr(mistral_voxtral.AudioModel, "converse", _fake_converse)

        model = AudioModel(_MODEL_ID)
        response = await model.stt(_FakeAudioContent(), "text", logprobs=False)  # type: ignore[arg-type]

        assert isinstance(response, Response)
        assert response.media_type == "text/plain; charset=utf-8"
        assert response.body == b"hello world"


class TestSttStreamAcceptsLogprobs:
    """Streaming ``stt_stream()``: event sequence, usage from stream metadata, logprobs.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/_default.py:AudioModel.stt_stream
    """

    @pytest.mark.parametrize("logprobs", [True, False])
    async def test_logprobs_is_always_none(
        self, monkeypatch: pytest.MonkeyPatch, logprobs: bool
    ) -> None:
        """Every streamed event reports ``logprobs=None``, requested or not.

        The stubbed stream yields one ``transcript.text.delta`` followed by the
        terminal ``transcript.text.done``, whose usage is built from the
        ConverseStream ``metadata.usage`` event. That usage carries no
        audio/text split, so ``input_token_details`` stays unset (issue #95).

        Ref: stdapi/types/openai_audio.py:TranscriptionTextDoneEvent
             https://github.com/stdapi-ai/stdapi.ai/issues/95
        """
        monkeypatch.setattr(
            mistral_voxtral.AudioModel, "converse_stream", _fake_converse_stream
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
