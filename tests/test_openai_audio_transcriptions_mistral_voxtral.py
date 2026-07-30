"""Tests for /v1/audio/transcriptions served by Mistral Voxtral on Bedrock.

Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
     stdapi/models/audio/mistral_voxtral.py:AudioModel
"""

from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"

VOXTRAL_ALL = (VOXTRAL_MINI,)
VOXTRAL_SAMPLE = (VOXTRAL_MINI,)

#: Words spoken by the ``sample_audio_mp3_file`` fixture ("This is a test.").
_SAMPLE_AUDIO_WORDS = ("test", "this")


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_official_api: bool) -> None:
    """Skip every test here: the Voxtral models have no official OpenAI equivalent."""
    if use_official_api:
        pytest.skip(
            "Mistral Voxtral models are not available on the official OpenAI API"
        )


class TestMistralVoxtralTranscriptions:
    """Transcription behavior specific to the Mistral Voxtral models.

    Voxtral is a Bedrock chat model driven through ``InvokeModel`` with a
    messages body, so it is billed in tokens (not audio seconds) and supports only
    the ``json`` and ``text`` response formats.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
         stdapi/models/audio/mistral_voxtral.py:AudioModel
    """

    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_basic_transcription_json(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``response_format=json`` returns the transcript plus token usage.

        Voxtral is billed per token, so the gateway emits the ``tokens`` usage variant
        built from the Bedrock ``usage`` block. This path reports ``audio_tokens: 0``
        and fills ``text_tokens`` from Bedrock's ``cached_tokens`` — inconsistent with
        the streaming path, which attributes every input token to audio (issue #95).
        Only the breakdown's consistency with ``input_tokens`` is asserted here so the
        test does not enshrine either mapping as intended.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
             https://github.com/stdapi-ai/stdapi.ai/issues/95
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
        )

        assert isinstance(response.text, str)
        text = response.text.strip()
        assert text, "Transcription returned an empty transcript"
        assert any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS), (
            f"Transcript does not match the sample audio: {text!r}"
        )

        # Validate usage information
        usage = response.usage
        assert usage is not None
        assert usage.type == "tokens", f"Unexpected usage variant: {usage!r}"
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens

        # Validate input token details
        details = usage.input_token_details
        assert details is not None, "Audio prompt token breakdown is missing"
        assert details.audio_tokens is not None
        assert details.text_tokens is not None
        assert details.audio_tokens + details.text_tokens <= usage.input_tokens, (
            f"Token breakdown exceeds the billed input tokens: {usage!r}"
        )

    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_transcription_text_format(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``response_format=text`` returns the bare transcript as a string.

        The model's message content is returned untouched for this format, so no
        usage or metadata envelope is produced.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="text",
        )

        assert isinstance(response, str)
        text = response.strip()
        assert text, "Transcription returned an empty transcript"
        assert any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS), (
            f"Transcript does not match the sample audio: {text!r}"
        )

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_with_temperature(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``temperature`` is accepted and forwarded to the Bedrock request.

        Unlike ``amazon.transcribe``, which rejects the parameter, Voxtral is a chat
        model whose request body carries a ``temperature`` field (defaulting to 0.0).

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
             stdapi/models/audio/mistral_voxtral.py:AudioModel._build_request
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            temperature=0.7,
        )

        assert isinstance(response.text, str)
        assert response.text.strip(), "Transcription returned an empty transcript"
        usage = response.usage
        assert usage is not None
        assert usage.type == "tokens"
        assert usage.input_tokens > 0

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_with_prompt(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``prompt`` is accepted and appended to the model's instruction text.

        ``amazon.transcribe`` rejects ``prompt`` outright; Voxtral instead receives it
        as extra text content after the built-in "Transcribe the audio." instruction.
        The transcript content is deliberately NOT asserted: with a prompt prepended
        Voxtral Mini frequently answers "The audio content is not available for
        transcription." instead of transcribing, so only acceptance and billing are
        stable enough to assert.

        Ref: https://developers.openai.com/api/docs/guides/speech-to-text#prompting
             stdapi/models/audio/__init__.py:AudioModelBase._built_prompt
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            prompt="This is a test audio file for transcription.",
        )

        assert isinstance(response.text, str)
        assert response.text.strip(), "Transcription returned an empty transcript"
        usage = response.usage
        assert usage is not None
        assert usage.type == "tokens"
        assert usage.output_tokens > 0
        # The prompt is billed on top of the audio, so the input is never audio-only.
        assert usage.input_tokens > 0

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_with_language(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``language`` is accepted and expressed as a natural-language hint.

        Voxtral has no language field: the gateway resolves the ISO-639-1 code to its
        language name and adds "The audio is excepted to be english language." to the
        prompt, so English audio still transcribes normally.

        Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
             stdapi/models/audio/__init__.py:AudioModelBase._built_prompt
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            language="en",
        )

        assert isinstance(response.text, str)
        text = response.text.strip()
        assert text, "Transcription returned an empty transcript"
        assert any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS), (
            f"Transcript does not match the sample audio: {text!r}"
        )

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_all_parameters(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``language``, ``prompt`` and ``temperature`` are accepted together.

        All three land in the same single Bedrock request — the first two inside the
        prompt text, the last as the sampling temperature — so combining them still
        yields a token-billed transcription.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/mistral_voxtral.py:AudioModel._build_request
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            language="en",
            prompt="Test audio transcription.",
            temperature=0.5,
        )

        assert isinstance(response.text, str)
        assert response.text.strip(), "Transcription returned an empty transcript"
        usage = response.usage
        assert usage is not None
        assert usage.type == "tokens", f"Unexpected usage variant: {usage!r}"
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens

    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_streaming_transcription(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """Streaming emits ``transcript.text.delta`` events then a final ``done`` event.

        Voxtral streams real Bedrock chunks, so several deltas may arrive; the closing
        ``transcript.text.done`` event repeats the concatenated deltas verbatim and is
        the only event carrying usage, taken from
        ``amazon-bedrock-invocationMetrics``.

        Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
             stdapi/models/audio/mistral_voxtral.py:AudioModel.stt_stream
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file), model=model_id, stream=True
        )

        chunks = []
        accumulated_text = ""
        has_delta_events = False
        has_done_event = False
        done_text = ""

        for chunk in response:
            chunks.append(chunk)

            if chunk.type == "transcript.text.delta":
                has_delta_events = True
                if chunk.delta:
                    accumulated_text += chunk.delta
                    assert chunk.delta.strip()

            elif chunk.type == "transcript.text.done":
                has_done_event = True
                done_text = chunk.text

                usage = chunk.usage
                assert usage is not None, "Done event carries no usage"
                assert usage.input_tokens > 0
                assert usage.output_tokens > 0
                assert usage.total_tokens == usage.input_tokens + usage.output_tokens
                assert usage.input_token_details is not None
                # The streaming path attributes every input token to audio, while the
                # non-streaming path reports audio_tokens=0 (issue #95).
                assert usage.input_token_details.audio_tokens == usage.input_tokens
                assert usage.input_token_details.text_tokens == 0

        assert len(chunks) > 0
        assert has_delta_events
        assert has_done_event
        assert chunks[-1].type == "transcript.text.done", (
            f"Stream does not end with the done event: {chunks[-1].type}"
        )
        assert accumulated_text.strip()
        assert done_text == accumulated_text, (
            "Done event text differs from the concatenated deltas: "
            f"{done_text!r} != {accumulated_text!r}"
        )

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_invalid_audio_file(self, openai_client: OpenAI, model_id: str) -> None:
        """A non-audio upload is rejected as a 400 ``invalid_request_error``.

        The gateway does not sniff the media itself: the ``text/plain`` upload becomes
        an ``input_audio`` block with format ``plain``, which Bedrock rejects with a
        ``ValidationException`` mapped to 400.

        Ref: stdapi/models/audio/mistral_voxtral.py:AudioModel._build_request
             stdapi/aws_bedrock.py:AWS_ERROR_MAP
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.transcriptions.create(
                file=("test.txt", b"This is not an audio file"), model=model_id
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["message"].strip(), "Error envelope carries no message"

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_verbose_json_unsupported(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``response_format=verbose_json`` is rejected with 400 for Voxtral.

        Voxtral returns plain text with no timing information, so the model declares
        only ``{"json", "text"}`` as supported formats and the request is refused
        before Bedrock is called.

        Ref: https://stdapi.ai/api_openai_audio_transcriptions/
             stdapi/models/audio/__init__.py:AudioModelBase._validate_response_formats
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.transcriptions.create(
                file=("test.mp3", sample_audio_mp3_file),
                model=model_id,
                response_format="verbose_json",
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "verbose_json" in error_body["message"], (
            f"Error does not name the refused format: {error_body['message']!r}"
        )
        assert "not supported" in error_body["message"]

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_logprobs_accepted_but_not_populated(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``include=["logprobs"]`` is accepted but no log probabilities are returned.

        ``amazon.transcribe`` rejects ``include`` outright; Voxtral accepts the request
        and forwards it, but Bedrock reports ``logprobs: null`` for every choice, so
        the field is always absent from the response.

        Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
             stdapi/models/audio/mistral_voxtral.py:AudioModel.stt
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            include=["logprobs"],
        )

        assert response.text.strip()
        assert response.logprobs is None

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_srt_format_unsupported(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """``response_format=srt`` is rejected with 400 for Voxtral.

        Subtitles are an Amazon Transcribe batch-job feature; a Bedrock chat model has
        no cue timings, so ``srt`` is outside Voxtral's supported format set.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/subtitles.html
             stdapi/models/audio/__init__.py:AudioModelBase._validate_response_formats
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.transcriptions.create(
                file=("test.mp3", sample_audio_mp3_file),
                model=model_id,
                response_format="srt",
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "srt" in error_body["message"], (
            f"Error does not name the refused format: {error_body['message']!r}"
        )
        assert "not supported" in error_body["message"]
