"""Unit tests for keyword and language context folding on Mistral Voxtral.

The gpt-transcribe context inputs (``keywords[]``, ``languages[]``) have no
native Voxtral parameter, so the gateway folds them into the transcription
prompt sent through the chat completions payload. Bedrock is stubbed; no AWS
call is made.

Ref: https://developers.openai.com/api/docs/guides/transcription
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
     stdapi/models/audio/mistral_voxtral.py:AudioModel._build_request
"""

from typing import Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.audio import mistral_voxtral
from stdapi.models.audio.mistral_voxtral import AudioModel

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


def _request_prompt(request: dict[str, Any]) -> str:
    """Return the text prompt carried by a built chat request.

    Args:
        request: The chat completions payload built by ``_build_request``.

    Returns:
        The text content part of the first message.
    """
    content = request["messages"][0]["content"]
    return str(next(part["text"] for part in content if part["type"] == "text"))


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
            "prompt_tokens_details": {"audio_tokens": 7, "cached_tokens": 5},
        },
    }


class TestContextFoldedIntoPrompt:
    """``keywords``/``languages`` become prompt context in the built request.

    Ref: https://developers.openai.com/api/docs/guides/transcription
         stdapi/models/audio/__init__.py:AudioModelBase._built_prompt
    """

    async def test_keywords_are_folded_into_the_prompt(self) -> None:
        """Every keyword literal appears in the transcription prompt.

        Ref: stdapi/models/audio/mistral_voxtral.py:AudioModel._build_request
        """
        request = await AudioModel(_MODEL_ID)._build_request(  # noqa: SLF001
            _FakeAudioContent(),  # type: ignore[arg-type]
            None,
            None,
            keywords=["Amoxicillin", "EBITDA"],
        )

        prompt = _request_prompt(dict(request))
        assert "Amoxicillin, EBITDA" in prompt

    async def test_languages_are_folded_into_the_prompt(self) -> None:
        """Expected languages appear as named language hints in the prompt.

        Ref: stdapi/models/audio/mistral_voxtral.py:AudioModel._build_request
        """
        request = await AudioModel(_MODEL_ID)._build_request(  # noqa: SLF001
            _FakeAudioContent(),  # type: ignore[arg-type]
            None,
            None,
            languages=["en", "fr"],
        )

        prompt = _request_prompt(dict(request))
        assert "english" in prompt
        assert "french" in prompt

    def test_single_entry_languages_matches_the_singular_language_hint(self) -> None:
        """``languages=["en"]`` builds the exact prompt ``language="en"`` builds.

        Ref: stdapi/models/audio/__init__.py:AudioModelBase._built_prompt
        """
        assert AudioModel._built_prompt(  # noqa: SLF001
            None, None, languages=["en"]
        ) == AudioModel._built_prompt(None, "en")  # noqa: SLF001

    async def test_stt_passes_context_to_the_invoked_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``stt()`` wires keywords and languages through to the Bedrock payload.

        Ref: stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
        """
        captured: dict[str, Any] = {}

        async def _fake_invoke(
            self: AudioModel,  # noqa: ARG001
            request: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            captured["request"] = request
            return InvokeResult(response=_fake_response())

        monkeypatch.setattr(mistral_voxtral.AudioModel, "invoke", _fake_invoke)

        response = await AudioModel(_MODEL_ID).stt(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "json",
            keywords=["Amoxicillin"],
            languages=["en", "fr"],
            logprobs=False,
        )

        assert response.text == "hello world"  # type: ignore[union-attr]
        prompt = _request_prompt(captured["request"])
        assert "Amoxicillin" in prompt
        assert "english" in prompt
        assert "french" in prompt
