"""Local OpenAI-compatible audio types."""

from typing import Annotated, Literal, Self

from pydantic import AliasChoices, BaseModel, Field, model_validator

from stdapi.api_errors import UnsupportedParameterError
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile  # noqa: TC001
from stdapi.types import BaseModelRequest, BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.openai import Auto

# Ref: openai.types.audio_response_format.AudioResponseFormat
AudioResponseFormat = Literal[
    "json", "text", "srt", "verbose_json", "vtt", "diarized_json"
]
TranslateAudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]

#: Subtitle formats for transcription/translation responses
AudioSubtitleFormat = Literal["srt", "vtt"]
SUBTITLE_FORMATS: set[AudioSubtitleFormat] = {"srt", "vtt"}

#: OpenAI voices and matching gender (True for Female, False elsewhere)
OPENAI_VOICES_FEMALE: dict[str, bool] = {
    "alloy": True,
    "ash": False,
    "ballad": True,
    "coral": True,
    "echo": False,
    "fable": True,
    "nova": True,
    "onyx": False,
    "sage": True,
    "shimmer": True,
    "verse": False,
}

# Ref: openai.types.audio.transcription_include.TranscriptionInclude
TranscriptionInclude = Literal["logprobs"]

AudioTimestampGranularities = Literal["word", "segment"]
AudioFileFormat = Literal["mp3", "ogg", "opus", "aac", "flac", "wav", "pcm"]


# Ref: openai.types.audio.transcription.Logprob
# Ref: openai.types.audio.transcription_text_delta_event.Logprob
# Ref: openai.types.audio.transcription_text_done_event.Logprob
class Logprob(BaseModel):
    """Log probability metadata for delta tokens."""

    token: str | None = Field(
        default=None, description="Token used to generate the log probability."
    )
    bytes: list[int | float] | None = Field(
        default=None, description="Bytes used to generate the log probability."
    )
    logprob: float | None = Field(
        default=None, description="Log probability of the token."
    )


# Ref: openai.types.audio.transcription_text_delta_event.TranscriptionTextDeltaEvent
class TranscriptionTextDeltaEvent(BaseModelResponse):
    """Streaming text delta event for transcriptions."""

    delta: str = Field(description="Transcribed text delta.")
    type: Literal["transcript.text.delta"] = Field(
        description="Event type. Always `transcript.text.delta`."
    )
    logprobs: list[Logprob] | None = Field(
        default=None,
        description="Log probabilities of the delta; included only if requested.",
    )


# Ref: openai.types.audio.transcription.UsageTokensInputTokenDetails
# Ref: openai.types.audio.transcription_text_done_event.UsageInputTokenDetails
class UsageInputTokenDetails(BaseModelResponse):
    """Details about the input tokens billed for this request."""

    audio_tokens: int | None = Field(
        default=None, description="Audio tokens billed for this request."
    )
    text_tokens: int | None = Field(
        default=None, description="Text tokens billed for this request."
    )


# Ref: openai.types.audio.transcription.UsageTokens
# Ref: openai.types.audio.transcription_text_done_event.Usage
# Ref: openai.types.audio.transcription_diarized.UsageTokens
class UsageTokens(BaseModelResponse):
    """Usage statistics for models billed by token usage."""

    input_tokens: int = Field(
        default=0, ge=0, description="Input tokens billed for this request."
    )
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated.")
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens used (input + output)."
    )
    type: Literal["tokens"] = Field(
        default="tokens",
        description="Usage object type. Always `tokens` for this variant.",
    )
    input_token_details: UsageInputTokenDetails | None = Field(
        default=None,
        description="Details about the input tokens billed for this request.",
    )


# Ref: openai.types.audio.transcription_text_done_event.TranscriptionTextDoneEvent
class TranscriptionTextDoneEvent(BaseModelResponse):
    """Streaming final done event for transcriptions."""

    text: str = Field(description="Transcribed text.")
    type: Literal["transcript.text.done"] = Field(
        description="Event type. Always `transcript.text.done`."
    )
    logprobs: list[Logprob] | None = Field(
        default=None,
        description="Log probabilities of individual tokens in the transcription.",
    )
    usage: UsageTokens | None = Field(
        default=None, description="Usage statistics for token-billed models."
    )


# Ref: openai.types.audio.transcription.UsageDuration
# Ref: openai.types.audio.transcription_verbose.Usage
# Ref: openai.types.audio.transcription_diarized.UsageDuration
class UsageDuration(BaseModelResponse):
    """Duration usage for models billed by audio duration."""

    seconds: float = Field(
        default=0, ge=0, description="Duration of the input audio in seconds."
    )
    type: Literal["duration"] = Field(
        description="Usage object type. Always `duration` for this variant."
    )


# Ref: openai.types.audio.transcription.Usage
# Ref: openai.types.audio.transcription_diarized.Usage
Usage = Annotated[UsageTokens | UsageDuration, Field(discriminator="type")]


# Ref: openai.types.audio.transcription.Transcription
class Transcription(BaseModelResponse):
    """Transcription response."""

    text: str = Field(description="Transcribed text.")
    logprobs: list[Logprob] | None = Field(
        default=None,
        description="Log probabilities of tokens; returned only with specific models when requested.",
    )
    usage: Usage | None = Field(
        default=None, description="Token or duration usage statistics."
    )


# Ref: openai.types.audio.transcription_segment.TranscriptionSegment
class TranscriptionSegment(BaseModelResponse):
    """Verbose JSON segment details."""

    id: int = Field(ge=0, description="Unique segment identifier.")
    avg_logprob: float = Field(
        description="Average logprob of the segment. Below -1 suggests logprobs failed."
    )
    compression_ratio: float = Field(
        ge=0, description="Compression ratio. Above 2.4 suggests compression failed."
    )
    end: float = Field(ge=0, description="End time of the segment in seconds.")
    no_speech_prob: float = Field(
        ge=0,
        description="Probability of no speech. Above 1.0 with avg_logprob below -1 indicates silence.",
    )
    seek: int = Field(ge=0, description="Seek offset of the segment.")
    start: float = Field(ge=0, description="Start time of the segment in seconds.")
    temperature: float = Field(
        description="Temperature parameter used for generating the segment."
    )
    text: str = Field(description="Text content of the segment.")
    tokens: list[int] = Field(description="Token IDs for the text content.")


# Ref: openai.types.audio.transcription_word.TranscriptionWord
class TranscriptionWord(BaseModelResponse):
    """Verbose JSON word details."""

    end: float = Field(ge=0, description="End time of the word in seconds.")
    start: float = Field(ge=0, description="Start time of the word in seconds.")
    word: str = Field(description="Text content of the word.")


# Ref: openai.types.audio.transcription_verbose.TranscriptionVerbose
class TranscriptionVerbose(BaseModelResponse):
    """Verbose JSON transcription response."""

    duration: float = Field(description="Duration of the input audio.")
    language: str = Field(description="Language of the input audio.")
    text: str = Field(description="Transcribed text.")
    segments: list[TranscriptionSegment] | None = Field(
        default=None,
        description="Transcribed text segments with corresponding details.",
    )
    usage: UsageDuration | None = Field(
        default=None, description="Usage statistics for duration-billed models."
    )
    words: list[TranscriptionWord] | None = Field(
        default=None, description="Extracted words with timestamps."
    )


# REF: openai.types.audio.transcription_diarized_segment.TranscriptionDiarizedSegment
class TranscriptionDiarizedSegment(BaseModelResponse):
    """A segment of diarized transcript text with speaker metadata."""

    id: str = Field(description="Unique segment identifier.")
    end: float = Field(ge=0, description="End timestamp of the segment in seconds.")
    speaker: str = Field(
        description="Speaker label: matches `known_speaker_names[]` if provided, "
        "otherwise sequential capital letters (`A`, `B`, ...)."
    )
    start: float = Field(ge=0, description="Start timestamp of the segment in seconds.")
    text: str = Field(description="Transcript text for this segment.")
    type: Literal["transcript.text.segment"] = Field(
        default="transcript.text.segment",
        description="Segment type. Always `transcript.text.segment`.",
    )


# REF: openai.types.audio.transcription_diarized.TranscriptionDiarized
class TranscriptionDiarized(BaseModelResponse):
    """Represents a diarized transcription response returned by the model, including the combined transcript and speaker-segment annotations."""

    duration: float = Field(ge=0, description="Duration of the input audio in seconds.")
    segments: list[TranscriptionDiarizedSegment] = Field(
        description="Transcript segments with timestamps and speaker labels."
    )
    task: Literal["transcribe"] = Field(
        default="transcribe", description="Task type. Always `transcribe`."
    )
    text: str = Field(description="Concatenated transcript for the entire audio input.")
    usage: Usage | None = Field(
        default=None, description="Token or duration usage statistics."
    )


# Ref: openai.types.audio.transcription_create_response.TranscriptionCreateResponse
TranscriptionCreateResponse = Transcription | TranscriptionVerbose


# Ref: openai.types.audio.transcription_create_params.ChunkingStrategyVadConfig
class ChunkingStrategyVadConfig(BaseModelRequest):
    """Manual server-side VAD chunking configuration."""

    type: Literal["server_vad"] = Field(
        description="Must be set to `server_vad` to enable manual chunking using server side VAD."
    )
    prefix_padding_ms: int | None = Field(
        default=None,
        ge=0,
        description="Amount of audio to include before the VAD detected speech (in milliseconds).",
    )
    silence_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Duration of silence to detect speech stop (in milliseconds). "
            "Shorter values respond faster but may cut in on short pauses."
        ),
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Sensitivity threshold (0.0 to 1.0) for voice activity detection. "
            "Higher values require louder audio to activate."
        ),
    )


# Ref: openai.types.audio.transcription_create_params.ChunkingStrategy
ChunkingStrategy = Auto | ChunkingStrategyVadConfig


# Ref: openai.types.audio.translation.Translation
class Translation(BaseModelResponse):
    """Translation response."""

    text: str = Field(description="Translated text.")


# Ref: openai.types.audio.translation_verbose.TranslationVerbose
class TranslationVerbose(BaseModelResponse):
    """Verbose JSON translation response."""

    duration: float = Field(description="Duration of the input audio.")
    language: str = Field(
        default="english", description="Output translation language (always `english`)."
    )
    text: str = Field(description="Translated text.")
    segments: list[TranscriptionSegment] | None = Field(
        default=None, description="Translated text segments with corresponding details."
    )


# Ref: openai.types.audio.translation_create_response.TranslationCreateResponse
TranslationCreateResponse = Translation | TranslationVerbose


# Speech SSE event types (following OpenAI pattern)
class SpeechAudioDeltaEvent(BaseModelResponse):
    """Speech audio delta event for streaming."""

    type: str = Field(default="speech.audio.delta", frozen=True)
    audio: str = Field(..., description="Base64-encoded audio chunk")


class SpeechUsage(BaseModelResponse):
    """Usage statistics for speech generation."""

    input_tokens: int = Field(..., description="Input tokens")
    output_tokens: int = Field(default=0, description="Output tokens")
    total_tokens: int = Field(..., description="Total tokens used")


class SpeechAudioDoneEvent(BaseModelResponse):
    """Speech audio done event for streaming."""

    type: str = Field(default="speech.audio.done", frozen=True)
    usage: SpeechUsage = Field(..., description="Usage statistics.")


# Ref: openai.types.audio.speech_create_params.SpeechCreateParams
class SpeechCreateParams(BaseModelRequestWithExtra, str_strip_whitespace=True):
    """Request model for text-to-speech generation."""

    input: str = Field(
        ...,
        validation_alias=AliasChoices("input", "Text"),
        min_length=1,
        description="Text to generate audio for. "
        "Amazon Polly models accept SSML documents.",
    )
    model: str = Field(
        default=SETTINGS.default_tts_model,
        validation_alias=AliasChoices("model", "Engine"),
        description="TTS model. "
        "Available: `amazon.polly-standard`, `amazon.polly-neural`, `amazon.polly-long-form`, `amazon.polly-generative`.",
    )
    voice: str = Field(
        default="alloy",
        validation_alias=AliasChoices("voice", "VoiceId"),
        description="Voice for audio generation. "
        "Supported voices vary by model and language.",
    )
    instructions: str | None = Field(
        default=None,
        description="Additional voice control instructions. "
        "Does not work with `amazon.polly-standard`, `amazon.polly-neural`, `amazon.polly-long-form`, or `amazon.polly-generative`.",
    )
    response_format: AudioFileFormat = Field(
        validation_alias=AliasChoices("response_format", "OutputFormat"),
        default="mp3",
        description="Audio format: `mp3`, `opus`, `ogg`, `aac`, `flac`, `wav`, or `pcm`.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.2,
        le=2.0,
        description="Audio speed. Range: `0.2` to `2.0`. Default: `1.0`.",
    )
    stream_format: Literal["audio", "sse"] = Field(
        default="audio",
        description="Streaming format: `sse` or `audio`. "
        "MCP tools default to `sse` for better client compatibility.",
    )

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible transcription options."""
        if self.input.startswith("<speak>") and "speed" in self.model_fields_set:
            msg = "speed is not supported for SSML input. In this case, set speed directly in SSML."
            raise ValueError(msg)
        return self


# Ref: openai.types.audio.transcription_create_params.TranscriptionCreateParams
class TranscriptionCreateParams(BaseModelRequestWithExtra, str_strip_whitespace=True):
    """Request model for audio transcription.

    Validates unsupported fields/values and incompatible combinations.
    """

    # file: handled in route
    model: str = Field(..., description="Transcription model to use.")
    chunking_strategy: ChunkingStrategy = Field(
        default="auto",
        description="Audio chunking: `auto` (VAD) or `server_vad` for manual tuning. "
        "server_vad is UNSUPPORTED.",
    )
    include: TranscriptionInclude | None = Field(
        default=None,
        description="Additional response info. "
        "`logprobs` returns token confidence; requires `response_format=json`.",
    )
    known_speaker_names: list[str] | None = Field(
        default=None,
        description="Speaker names matching `known_speaker_references[]` (e.g. `customer`, `agent`). "
        "UNSUPPORTED.",
    )
    known_speaker_references: list[str] | None = Field(
        default=None,
        description="Audio samples (data URLs, 2-10s) for known-speaker diarization. "
        "UNSUPPORTED.",
    )
    language: str | None = Field(
        default=None,
        description="Input audio language in ISO-639-1 (e.g. `en`). Improves accuracy and latency.",
    )
    prompt: str | None = Field(
        default=None,
        description="Text to guide model style or continue a previous segment. "
        "Should match the audio language.",
    )
    response_format: AudioResponseFormat = Field(
        default="json", description="Transcript output format."
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        description="Sampling temperature (`0` to `1`). "
        "Higher (e.g. `0.8`) is more random; lower (e.g. `0.2`) is more focused.",
    )
    timestamp_granularities: list[AudioTimestampGranularities] = Field(
        default_factory=list,
        description="Timestamp granularities: `word` and/or `segment`. "
        "Requires `response_format=verbose_json`.",
    )
    stream: bool | None = Field(
        default=False,
        description="Stream the response as server-sent events as it is generated.",
    )

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible transcription options.

        Rules implemented:
        - timestamp_granularities may only be used with response_format == 'verbose_json'.
        - chunking_strategy other than 'auto' is unsupported.
        - prompt is unsupported.
        - temperature values other than 0.0 are unsupported.
        """
        if self.timestamp_granularities and self.response_format != "verbose_json":
            msg = "timestamp_granularities requires response_format='verbose_json'."
            raise ValueError(msg)
        if isinstance(self.chunking_strategy, dict) or self.chunking_strategy != "auto":
            # Any explicit server_vad config or non-auto is unsupported
            param = "chunking_strategy"
            raise UnsupportedParameterError(param)
        return self


# Ref: openai.types.audio.translation_create_params.TranslationCreateParams
class TranslationCreateParams(BaseModelRequestWithExtra, str_strip_whitespace=True):
    """Request model for audio translation.

    Validates unsupported fields/values and incompatible combinations.
    """

    # file: handled in route
    model: str = Field(..., description="Transcription model to use.")
    prompt: str | None = Field(
        default=None,
        description="Text to guide model style or continue a previous segment. "
        "Should be in English.",
    )
    response_format: TranslateAudioResponseFormat = Field(
        default="json", description="Output format."
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (`0` to `1`). "
        "Higher values like `0.8` will make the output more random, while lower values like "
        "`0.2` will make it more focused and deterministic.",
    )


#: Shared description for the ``file`` field in audio JSON body models.
_AUDIO_FILE_FIELD_DESCRIPTION = (
    "Audio file: base64 string, data URI, HTTPS URL, S3 URI, or Files API reference. "
    "Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm."
)


class AudioTranscriptionJsonBody(TranscriptionCreateParams):
    """Request body for audio transcription via ``application/json``.

    Alternative to ``multipart/form-data`` for MCP tools and HTTP clients that
    cannot construct multipart requests. The ``file`` field accepts a base64
    string, a data URI, an HTTPS URL, or an S3 URI.
    """

    file: InputFile = Field(description=_AUDIO_FILE_FIELD_DESCRIPTION)


class AudioTranslationJsonBody(TranslationCreateParams):
    """Request body for audio translation via ``application/json``.

    Alternative to ``multipart/form-data`` for MCP tools and HTTP clients that
    cannot construct multipart requests. The ``file`` field accepts a base64
    string, a data URI, an HTTPS URL, or an S3 URI.
    """

    file: InputFile = Field(description=_AUDIO_FILE_FIELD_DESCRIPTION)
