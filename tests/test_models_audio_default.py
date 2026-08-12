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
from stdapi.aws_bedrock import BEDROCK_BODY_SIZE_LIMIT, MIME_TYPES_TO_AUDIO_TYPE
from stdapi.models import ModelDetails, _compute_model_capabilities
from stdapi.models.audio import _default, get_audio_model
from stdapi.models.audio._default import CONVERSE_AUDIO_FORMATS, AudioModel
from stdapi.types.openai_audio import TranscriptionTextDoneEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: A synthetic Converse-capable SPEECH-input model with no dedicated audio module.
_MODEL_ID = "synthetic.speech-to-text-v1:0"

#: A speech model Converse cannot serve; it is served by its own class instead.
_NOVA_SONIC_ID = "amazon.nova-2-sonic-v1:0"

#: Chunks the abort test's encode may yield before the test fails on its own.
_ABORT_CHUNK_BUDGET = 64


class _FakeAudioContent:
    """Minimal ``InputFile`` stand-in with a configurable content type."""

    def __init__(
        self,
        media_type: str = "audio",
        file_format: str = "mp3",
        data: bytes = b"fake",
        reported_size: int | None = None,
    ) -> None:
        """Store the reported content type parts, the payload and its size.

        Args:
            media_type: Media type reported by the content sniffer.
            file_format: Format subtype reported by the content sniffer.
            data: The payload returned by ``to_bytes``.
            reported_size: Size to report instead of the payload length, as a
                chunked response or a length-less upload reports zero.
        """
        self._media_type = media_type
        self._file_format = file_format
        self._data = data
        self._reported_size = reported_size

    async def get_content_type_tuple(self) -> tuple[str, str]:
        """Report the configured content type."""
        return (self._media_type, self._file_format)

    async def get_size(self) -> int:
        """Report the payload size the source knows before reading the body."""
        if self._reported_size is not None:
            return self._reported_size
        return len(self._data)

    async def to_bytes(self) -> bytes:
        """Return the configured audio payload."""
        return self._data


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

        content = request["messages"][0]["content"]
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


class TestNonConverseSpeechModels:
    """The Converse default refuses speech models Converse cannot serve.

    Those models reach the API through their own class instead (see
    ``tests/test_models_nova_sonic.py``); this default is what protects any that
    have none from a Converse call that is doomed before it is sent.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-features.html
         stdapi/models/audio/_default.py:AudioModel._validate_converse_supported
    """

    async def test_stt_is_rejected(self) -> None:
        """``stt()`` raises a clear error instead of sending a doomed Converse call."""
        with pytest.raises(ApiError, match="not supported"):
            await AudioModel(_NOVA_SONIC_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )

    async def test_stt_stream_is_rejected(self) -> None:
        """``stt_stream()`` raises the same clear error."""
        with pytest.raises(ApiError, match="not supported"):
            async for _ in AudioModel(_NOVA_SONIC_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                logprobs=False,
            ):
                pass

    async def test_stt_translate_is_rejected(self) -> None:
        """``stt_translate()`` raises the same clear error."""
        with pytest.raises(ApiError, match="not supported"):
            await AudioModel(_NOVA_SONIC_ID).stt_translate(
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
        # type-ignore: only botocore's StringShape carries .enum, shape_for is untyped.
        assert frozenset(shape.enum) == CONVERSE_AUDIO_FORMATS  # type: ignore[attr-defined]

    def test_every_mime_alias_targets_a_converse_audio_format(self) -> None:
        """MIME aliases resolve to enum members, so they never trigger a transcode.

        Ref: stdapi/aws_bedrock.py:MIME_TYPES_TO_AUDIO_TYPE
        """
        assert set(MIME_TYPES_TO_AUDIO_TYPE.values()) <= CONVERSE_AUDIO_FORMATS

    @pytest.mark.parametrize(
        ("media_type", "file_format", "expected"),
        [
            ("audio", "mp3", "mp3"),
            ("audio", "flac", "flac"),
            ("audio", "mpeg", "mp3"),
            ("audio", "x-wav", "wav"),
            ("audio", "x-m4a", "mp4"),
            ("video", "webm", "webm"),
            # libmagic labels both .mka and .mkv "video/x-matroska", and both
            # are Converse audio formats: they must not be transcoded.
            ("video", "x-matroska", "mkv"),
            ("audio", "x-matroska", "mkv"),
            # libmagic labels raw ADTS AAC streams "audio/x-hx-aac-adts".
            ("audio", "x-hx-aac-adts", "aac"),
        ],
    )
    async def test_supported_formats_pass_through_inline(
        self, media_type: str, file_format: str, expected: str
    ) -> None:
        """Formats in the enum are sent as inline bytes without transcoding."""
        block = await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
            _FakeAudioContent(media_type, file_format)  # type: ignore[arg-type]
        )

        assert block["audio"]["format"] == expected
        assert block["audio"]["source"] == {"bytes": b"fake"}

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
        assert block["audio"]["format"] == "flac"
        assert block["audio"]["source"] == {"bytes": b"transcoded"}

    async def test_legacy_audio_seen_as_video_is_normalized_to_flac(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A WMA upload transcodes instead of being rejected as non-audio.

        libmagic identifies the ASF container by its header GUID and reports
        ``video/x-ms-asf`` whether or not the payload carries a video track, so
        gating the fallback on ``media_type == "audio"`` would make WMA — the
        flagship legacy case for this fallback — unreachable.
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
            _FakeAudioContent("video", "x-ms-asf")  # type: ignore[arg-type]
        )

        assert captured == {"input": b"fake", "output_format": "flac"}
        assert block["audio"]["format"] == "flac"
        assert block["audio"]["source"] == {"bytes": b"transcoded"}

    @pytest.mark.parametrize(
        ("media_type", "file_format"),
        [("application", "octet-stream"), ("application", "pdf"), ("text", "plain")],
    )
    async def test_non_audio_upload_is_rejected_with_the_accepted_list(
        self, media_type: str, file_format: str
    ) -> None:
        """Content that is neither audio nor video gets a 400 listing accepted formats."""
        with pytest.raises(ApiError, match="Accepted formats") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent(media_type, file_format)  # type: ignore[arg-type]
            )

        assert exc_info.value.status == 400
        assert "mp3" in str(exc_info.value)
        assert "wav" in str(exc_info.value)


class TestFallbackTranscodeFailure:
    """An undecodable upload is the client's problem, not a server error.

    Ref: stdapi/media.py:encode_audio_stream
         stdapi/models/audio/_default.py:AudioModel._audio_content_block
    """

    async def test_encode_failure_becomes_a_400_naming_the_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ffmpeg failing to decode the upload surfaces as a 400, not a 500.

        A container with no audio track, or a codec the ffmpeg build cannot
        decode, makes the pipeline exit nonzero and raise a 500; at this point
        in the flow the payload came from the client, so the status is 400.
        """

        async def _failing_encode(
            stream: AsyncGenerator[bytes],  # noqa: ARG001
            output_format: str,
        ) -> AsyncGenerator[bytes]:
            msg = f"Failed to encode the audio to '{output_format}'."
            raise ApiError(msg, status=500)
            yield b""  # pragma: no cover - unreachable, keeps this a generator

        monkeypatch.setattr(_default, "encode_audio_stream", _failing_encode)

        with pytest.raises(ApiError, match="could not be decoded as audio") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent("video", "x-ms-asf")  # type: ignore[arg-type]
            )

        assert exc_info.value.status == 400
        assert "x-ms-asf" in str(exc_info.value)

    async def test_encode_timeout_stays_a_504(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stalled ffmpeg keeps its gateway-timeout status instead of becoming a 400."""

        async def _timing_out_encode(
            stream: AsyncGenerator[bytes],  # noqa: ARG001
            output_format: str,
        ) -> AsyncGenerator[bytes]:
            msg = f"Timed out encoding the audio to '{output_format}'."
            raise ApiError(msg, status=504)
            yield b""  # pragma: no cover - unreachable, keeps this a generator

        monkeypatch.setattr(_default, "encode_audio_stream", _timing_out_encode)

        with pytest.raises(ApiError, match="Timed out") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent("audio", "amr")  # type: ignore[arg-type]
            )

        assert exc_info.value.status == 504


class TestInlineBodySizeGuard:
    """Inline Converse audio is bounded by the Bedrock request body limit.

    The audio travels as inline bytes in the Converse request, and FLAC is
    roughly PCM-sized, so a transcoded upload can outgrow what Bedrock accepts
    and come back as an opaque upstream ValidationException.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_AudioSource.html
         stdapi/models/audio/_default.py:AudioModel._check_inline_size
    """

    async def test_oversized_native_format_is_rejected(self) -> None:
        """A natively supported upload over the limit gets an actionable 400."""
        with pytest.raises(ApiError, match="too large") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent(  # type: ignore[arg-type]
                    "audio", "mp3", b"\0" * (BEDROCK_BODY_SIZE_LIMIT + 1)
                )
            )

        assert exc_info.value.status == 400
        assert "shorter file" in str(exc_info.value)

    async def test_oversized_transcode_result_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A FLAC conversion that outgrows the limit is caught before the Converse call."""

        async def _fat_encode(
            stream: AsyncGenerator[bytes],  # noqa: ARG001
            output_format: str,  # noqa: ARG001
        ) -> AsyncGenerator[bytes]:
            yield b"\0" * (BEDROCK_BODY_SIZE_LIMIT + 1)

        monkeypatch.setattr(_default, "encode_audio_stream", _fat_encode)

        with pytest.raises(ApiError, match="too large") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent("audio", "amr")  # type: ignore[arg-type]
            )

        assert exc_info.value.status == 400
        assert "FLAC" in str(exc_info.value)

    async def test_oversized_payload_reporting_no_size_is_rejected(self) -> None:
        """A source that under-reports its size is caught on the bytes read.

        An HTTP source answering with ``Transfer-Encoding: chunked`` and an
        upload without a declared length both report zero, so the pre-read
        check alone would hand Bedrock a body it refuses.
        """
        with pytest.raises(ApiError, match="too large") as exc_info:
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent(  # type: ignore[arg-type]
                    "audio",
                    "mp3",
                    b"\0" * (BEDROCK_BODY_SIZE_LIMIT + 1),
                    reported_size=0,
                )
            )

        assert exc_info.value.status == 400

    async def test_transcode_stops_once_the_result_cannot_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The encode is abandoned at the limit rather than buffered whole.

        A long audio track transcodes to far more FLAC than Converse accepts,
        so consuming the whole stream to then reject it would let an upload
        choose how much memory the server spends.
        """
        chunks_yielded = 0
        closed = False
        chunk = b"\0" * (BEDROCK_BODY_SIZE_LIMIT // 4)

        async def _endless_encode(
            stream: AsyncGenerator[bytes],  # noqa: ARG001
            output_format: str,  # noqa: ARG001
        ) -> AsyncGenerator[bytes]:
            nonlocal chunks_yielded, closed
            try:
                # Bounded so a lost abort fails this test instead of hanging it.
                for _ in range(_ABORT_CHUNK_BUDGET):
                    chunks_yielded += 1
                    yield chunk
            finally:
                closed = True

        monkeypatch.setattr(_default, "encode_audio_stream", _endless_encode)

        with pytest.raises(ApiError, match="too large"):
            await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
                _FakeAudioContent("audio", "amr")  # type: ignore[arg-type]
            )

        assert chunks_yielded == 5, "the encode ran past the first oversized chunk"
        assert closed, "the ffmpeg pipeline was left running"

    async def test_payload_at_the_limit_is_accepted(self) -> None:
        """The limit itself is inclusive: only what exceeds it is refused."""
        block = await AudioModel(_MODEL_ID)._audio_content_block(  # noqa: SLF001
            _FakeAudioContent(  # type: ignore[arg-type]
                "audio", "mp3", b"\0" * BEDROCK_BODY_SIZE_LIMIT
            )
        )

        assert len(block["audio"]["source"]["bytes"]) == BEDROCK_BODY_SIZE_LIMIT  # type: ignore[arg-type]


class TestConverseStreamEvents:
    """``stt_stream()`` maps Converse stream events onto the OpenAI SSE events.

    Ref: https://developers.openai.com/api/docs/api-reference/audio/create-transcription
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
         stdapi/models/audio/_default.py:AudioModel.stt_stream
    """

    @staticmethod
    def _patch_stream(
        monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
    ) -> None:
        """Make ``converse_stream`` replay *events* instead of calling Bedrock."""

        async def _fake_converse_stream(
            self: AudioModel,  # noqa: ARG001
            _request: object,
        ) -> dict[str, Any]:
            async def _stream() -> AsyncGenerator[dict[str, Any]]:
                for event in events:
                    yield event

            return {"stream": _stream()}

        monkeypatch.setattr(
            _default.AudioModel, "converse_stream", _fake_converse_stream
        )

    async def test_deltas_stream_then_a_done_event_carries_the_full_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each text delta is forwarded, and the done event concatenates them all.

        A client rendering only the deltas and a client reading only the final
        event must end up with the same transcript.
        """
        self._patch_stream(
            monkeypatch,
            [
                {"contentBlockDelta": {"delta": {"text": "hello "}}},
                # A non-text block (e.g. reasoning) must not become a delta.
                {"contentBlockDelta": {"delta": {"reasoningContent": {}}}},
                {"contentBlockDelta": {"delta": {"text": "world"}}},
                {
                    "metadata": {
                        "usage": {
                            "inputTokens": 10,
                            "outputTokens": 2,
                            "totalTokens": 12,
                        }
                    }
                },
            ],
        )

        events = [
            event
            async for event in AudioModel(_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                logprobs=False,
            )
        ]

        assert [event.delta for event in events[:-1]] == ["hello ", "world"]  # type: ignore[union-attr]
        done = events[-1]
        assert isinstance(done, TranscriptionTextDoneEvent)
        assert done.text == "hello world"
        assert done.usage is not None
        assert done.usage.model_dump() == {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "type": "tokens",
            # Converse reports no audio/text input split (issue #95).
            "input_token_details": None,
        }
        assert all(event.logprobs is None for event in events)

    async def test_stream_without_usage_metadata_reports_no_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stream carrying no ``metadata.usage`` leaves the done event's usage unset.

        Reporting zeros instead would bill-report a request as free.
        """
        self._patch_stream(
            monkeypatch, [{"contentBlockDelta": {"delta": {"text": "hi"}}}]
        )

        events = [
            event
            async for event in AudioModel(_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                logprobs=False,
            )
        ]

        done = events[-1]
        assert isinstance(done, TranscriptionTextDoneEvent)
        assert done.text == "hi"
        assert done.usage is None


class TestUnsupportedResponseFormats:
    """The Converse default serves ``json``/``text`` only, and says so.

    Subtitle and verbose formats are built from timestamps the Converse API
    never returns, so they are refused up front instead of returning an
    empty-timestamp response.

    Ref: https://developers.openai.com/api/docs/api-reference/audio/create-transcription
         stdapi/models/audio/__init__.py:AudioModelBase._validate_response_formats
    """

    @pytest.mark.parametrize(
        "response_format", ["srt", "vtt", "verbose_json", "diarized_json"]
    )
    async def test_transcription_rejects_timestamped_formats(
        self, response_format: str
    ) -> None:
        """An unsupported ``response_format`` fails with a 400 naming it."""
        with pytest.raises(ApiError, match="is not supported") as exc_info:
            await AudioModel(_MODEL_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                response_format,  # type: ignore[arg-type]
                logprobs=False,
            )

        assert exc_info.value.status == 400
        assert response_format in str(exc_info.value)

    async def test_translation_rejects_timestamped_formats(self) -> None:
        """``stt_translate()`` applies the same response-format allowlist."""
        with pytest.raises(ApiError, match="is not supported"):
            await AudioModel(_MODEL_ID).stt_translate(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "srt",
                None,
            )

    def test_supported_response_formats_are_json_and_text(self) -> None:
        """The allowlist itself is pinned: widening it needs the formats built."""
        assert frozenset({"json", "text"}) == AudioModel.SUPPORTED_RESPONSES_FORMATS


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
