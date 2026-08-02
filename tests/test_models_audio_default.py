"""Unit tests for the generic Converse speech-to-text default model.

Any Bedrock model with a SPEECH input modality and Converse support can
transcribe through this default without a dedicated audio module. Bedrock is
stubbed; no AWS call is made.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
     stdapi/models/audio/_default.py:AudioModel
"""

from typing import TYPE_CHECKING, Any

import pytest
from botocore.session import get_session
from starlette.responses import Response

import stdapi.routes.openai_audio_transcriptions
import stdapi.routes.openai_audio_translations  # noqa: F401  (registers the STT_TRANSLATE route capability)
from stdapi.api_errors import ApiError
from stdapi.models import ModelDetails, _compute_model_capabilities
from stdapi.models.audio import _default, get_audio_model
from stdapi.models.audio._default import CONVERSE_AUDIO_FORMATS, AudioModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: A synthetic Converse-capable SPEECH-input model with no dedicated audio module.
_MODEL_ID = "synthetic.speech-to-text-v1:0"

#: The speech model without Converse support (bidirectional streaming API only).
_NOVA_SONIC_ID = "amazon.nova-2-sonic-v1:0"


class _FakeAudioContent:
    """Minimal ``InputFile`` stand-in with a configurable content type."""

    def __init__(self, media_type: str = "audio", file_format: str = "mp3") -> None:
        """Store the reported content type parts."""
        self._media_type = media_type
        self._file_format = file_format

    async def get_content_type_tuple(self) -> tuple[str, str]:
        """Report the configured content type."""
        return (self._media_type, self._file_format)

    async def to_bytes(self) -> bytes:
        """Return a fixed audio payload."""
        return b"fake"


def _fake_converse_response(text: str = "hello world") -> dict[str, Any]:
    """Build a minimal Bedrock Converse response payload."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
    }


def _model_details(
    model_id: str, input_modalities: list[str], output_modalities: list[str]
) -> ModelDetails:
    """Build minimal model details for capability computation."""
    return ModelDetails(
        id=model_id,
        name=model_id,
        provider="Test",
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        regions=["us-west-2"],
    )


class TestConverseRequestShape:
    """The built Converse request: block order, prompt and inference config.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_AudioBlock.html
         stdapi/models/audio/_default.py:AudioModel._build_request
    """

    async def test_audio_block_precedes_the_text_prompt(self) -> None:
        """The audio block comes first; text-first makes speech models ignore the audio.

        Live-probed on mistral.voxtral-mini-3b-2507: with the text prompt block
        before the audio block, the model ignores the audio entirely. The order
        is load-bearing and must never be swapped.

        Ref: stdapi/models/audio/_default.py:AudioModel._build_request
        """
        request = await AudioModel(_MODEL_ID)._build_request(  # noqa: SLF001
            _FakeAudioContent(),  # type: ignore[arg-type]
            None,
            None,
        )

        content = request["messages"][0]["content"]  # type: ignore[index]
        assert len(content) == 2
        assert "audio" in content[0]
        assert "text" in content[1]

    async def test_temperature_defaults_to_zero(self) -> None:
        """An unset temperature is pinned to 0.0 in the inference config."""
        request = await AudioModel(_MODEL_ID)._build_request(  # noqa: SLF001
            _FakeAudioContent(),  # type: ignore[arg-type]
            None,
            None,
        )

        assert request["inferenceConfig"] == {"temperature": 0.0}

    async def test_temperature_is_forwarded_to_the_inference_config(self) -> None:
        """An explicit temperature is carried by ``inferenceConfig``."""
        request = await AudioModel(_MODEL_ID)._build_request(  # noqa: SLF001
            _FakeAudioContent(),  # type: ignore[arg-type]
            None,
            0.7,
        )

        assert request["inferenceConfig"] == {"temperature": 0.7}


class TestGenericDefaultResolution:
    """Models without a dedicated audio module resolve to the Converse default.

    Ref: stdapi/models/__init__.py:get_model
         stdapi/models/audio/_default.py:AudioModel
    """

    def test_unmatched_speech_model_resolves_to_the_default(self) -> None:
        """``get_audio_model`` falls back to the Converse default class."""
        model = get_audio_model(_MODEL_ID)
        assert type(model) is AudioModel

    async def test_default_transcribes_via_converse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A synthetic SPEECH+Converse model transcribes through the default path."""
        captured: dict[str, Any] = {}

        async def _fake_converse(
            self: AudioModel,  # noqa: ARG001
            request: Any,  # noqa: ANN401
        ) -> dict[str, Any]:
            captured["request"] = request
            return _fake_converse_response()

        monkeypatch.setattr(_default.AudioModel, "converse", _fake_converse)

        response = await get_audio_model(_MODEL_ID).stt(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "json",
            logprobs=False,
        )

        assert response.text == "hello world"  # type: ignore[union-attr]
        assert "audio" in captured["request"]["messages"][0]["content"][0]

    def test_speech_model_without_audio_class_advertises_transcription(self) -> None:
        """A Converse SPEECH-input model advertises the transcription routes.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
        """
        routes, _tools = _compute_model_capabilities(
            _MODEL_ID, _model_details(_MODEL_ID, ["SPEECH", "TEXT"], ["TEXT"])
        )

        assert "/v1/audio/transcriptions" in routes
        assert "/v1/audio/translations" in routes


class TestNovaSonicExcluded:
    """nova-2-sonic has SPEECH input but no Converse API: never advertised or routed.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-features.html
         stdapi/models/audio/_default.py:AudioModel._validate_converse_supported
    """

    def test_nova_sonic_is_not_advertised_for_transcription(self) -> None:
        """Capability computation skips the STT default for non-Converse speech models.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
        """
        routes, _tools = _compute_model_capabilities(
            _NOVA_SONIC_ID, _model_details(_NOVA_SONIC_ID, ["SPEECH", "TEXT"], ["TEXT"])
        )

        assert "/v1/audio/transcriptions" not in routes
        assert "/v1/audio/translations" not in routes

    async def test_nova_sonic_stt_is_rejected(self) -> None:
        """``stt()`` raises a clear error instead of sending a doomed Converse call."""
        with pytest.raises(ApiError, match="not supported"):
            await get_audio_model(_NOVA_SONIC_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )

    async def test_nova_sonic_stt_stream_is_rejected(self) -> None:
        """``stt_stream()`` raises the same clear error."""
        with pytest.raises(ApiError, match="not supported"):
            async for _ in get_audio_model(_NOVA_SONIC_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                logprobs=False,
            ):
                pass

    async def test_nova_sonic_stt_translate_is_rejected(self) -> None:
        """``stt_translate()`` raises the same clear error."""
        with pytest.raises(ApiError, match="not supported"):
            await get_audio_model(_NOVA_SONIC_ID).stt_translate(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                None,
            )


class TestAudioFormatMapping:
    """Uploaded MIME types map onto the Converse ``AudioFormat`` enum.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_AudioBlock.html
         stdapi/models/audio/_default.py:AudioModel._audio_content_block
    """

    def test_accepted_set_matches_the_installed_botocore_enum(self) -> None:
        """The hardcoded accepted set stays in sync with botocore's enum."""
        shape = (
            get_session().get_service_model("bedrock-runtime").shape_for("AudioFormat")
        )
        assert frozenset(shape.enum) == CONVERSE_AUDIO_FORMATS

    @pytest.mark.parametrize(
        ("media_type", "file_format", "expected"),
        [
            ("audio", "mp3", "mp3"),
            ("audio", "flac", "flac"),
            ("audio", "mpeg", "mp3"),
            ("audio", "x-wav", "wav"),
            ("audio", "x-m4a", "mp4"),
            ("video", "webm", "webm"),
        ],
    )
    async def test_supported_formats_pass_through_inline(
        self, media_type: str, file_format: str, expected: str
    ) -> None:
        """Formats in the enum are sent as inline bytes without transcoding."""
        block = await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
            _FakeAudioContent(media_type, file_format)  # type: ignore[arg-type]
        )

        assert block["audio"]["format"] == expected  # type: ignore[typeddict-item]
        assert block["audio"]["source"] == {"bytes": b"fake"}  # type: ignore[typeddict-item]

    async def test_unsupported_audio_format_is_normalized_to_flac(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audio outside the enum goes through the ffmpeg pipeline to FLAC.

        FLAC is lossless and its encoder/muxer are part of the minimal ffmpeg
        build shipped in the container image, unlike MP3 which would require
        linking LAME.
        """
        captured: dict[str, Any] = {}

        async def _fake_encode(
            stream: AsyncGenerator[bytes], output_format: str
        ) -> AsyncGenerator[bytes]:
            captured["input"] = b"".join([chunk async for chunk in stream])
            captured["output_format"] = output_format
            yield b"transcoded"

        monkeypatch.setattr(_default, "encode_audio_stream", _fake_encode)

        block = await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
            _FakeAudioContent("audio", "amr")  # type: ignore[arg-type]
        )

        assert captured == {"input": b"fake", "output_format": "flac"}
        assert block["audio"]["format"] == "flac"  # type: ignore[typeddict-item]
        assert block["audio"]["source"] == {"bytes": b"transcoded"}  # type: ignore[typeddict-item]

    async def test_non_audio_upload_is_rejected_with_the_accepted_list(self) -> None:
        """Non-audio content outside the enum gets a 400 listing accepted formats."""
        with pytest.raises(ApiError, match="Accepted formats") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent("application", "octet-stream")  # type: ignore[arg-type]
            )

        assert "mp3" in str(exc_info.value)
        assert "wav" in str(exc_info.value)


class TestTranslatePromptPath:
    """``stt_translate()`` folds the translation directive into the prompt.

    Ref: https://developers.openai.com/api/docs/api-reference/audio/create-translation
         stdapi/models/audio/_default.py:AudioModel.stt_translate
    """

    async def test_translate_adds_the_translation_directive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Converse prompt carries both transcription and translation directives."""
        captured: dict[str, Any] = {}

        async def _fake_converse(
            self: AudioModel,  # noqa: ARG001
            request: Any,  # noqa: ANN401
        ) -> dict[str, Any]:
            captured["request"] = request
            return _fake_converse_response("bonjour becomes hello")

        monkeypatch.setattr(_default.AudioModel, "converse", _fake_converse)

        response = await AudioModel(_MODEL_ID).stt_translate(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "json",
            None,
        )

        assert response.text == "bonjour becomes hello"  # type: ignore[union-attr]
        content = captured["request"]["messages"][0]["content"]
        assert "audio" in content[0]
        prompt = content[1]["text"]
        assert AudioModel.TRANSCRIPTION_PROMPT in prompt
        assert AudioModel.TRANSLATION_PROMPT in prompt

    async def test_translate_text_format_is_a_raw_plain_text_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``response_format=text`` returns the bare translated transcript."""

        async def _fake_converse(self: AudioModel, _request: object) -> dict[str, Any]:  # noqa: ARG001
            return _fake_converse_response()

        monkeypatch.setattr(_default.AudioModel, "converse", _fake_converse)

        response = await AudioModel(_MODEL_ID).stt_translate(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "text",
            None,
        )

        assert isinstance(response, Response)
        assert response.body == b"hello world"
