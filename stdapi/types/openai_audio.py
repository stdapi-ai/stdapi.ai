"""Local OpenAI-compatible audio types."""

from typing import Annotated, Literal, Self

from pydantic import AliasChoices, BaseModel, Field, model_validator

from stdapi.openai_exceptions import OpenaiUnsupportedParameterError
from stdapi.types import BaseModelRequest, BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.openai import Auto

# Ref: openai.types.audio_response_format.AudioResponseFormat
AudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]

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
        default=None,
        description="The token that was used to generate the log probability.",
    )
    bytes: list[int | float] | None = Field(
        default=None,
        description="The bytes that were used to generate the log probability.",
    )
    logprob: float | None = Field(
        default=None, description="The log probability of the token."
    )


# Ref: openai.types.audio.transcription_text_delta_event.TranscriptionTextDeltaEvent
class TranscriptionTextDeltaEvent(BaseModelResponse):
    """Streaming text delta event for transcriptions."""

    delta: str = Field(description="The text delta that was additionally transcribed.")
    type: Literal["transcript.text.delta"] = Field(
        description="The type of the event. Always `transcript.text.delta`."
    )
    logprobs: list[Logprob] | None = Field(
        default=None,
        description=(
            "The log probabilities of the delta. Only included if requested by the client."
        ),
    )


# Ref: openai.types.audio.transcription.UsageTokensInputTokenDetails
# Ref: openai.types.audio.transcription_text_done_event.UsageInputTokenDetails
class UsageInputTokenDetails(BaseModelResponse):
    """Details about the input tokens billed for this request."""

    audio_tokens: int | None = Field(
        default=None, description="Number of audio tokens billed for this request."
    )
    text_tokens: int | None = Field(
        default=None, description="Number of text tokens billed for this request."
    )


# Ref: openai.types.audio.transcription.UsageTokens
# Ref: openai.types.audio.transcription_text_done_event.Usage
class UsageTokens(BaseModelResponse):
    """Usage statistics for models billed by token usage."""

    input_tokens: int = Field(
        default=0, ge=0, description="Number of input tokens billed for this request."
    )
    output_tokens: int = Field(
        default=0, ge=0, description="Number of output tokens generated."
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Total number of tokens used (input + output)."
    )
    type: Literal["tokens"] = Field(
        default="tokens",
        description="The type of the usage object. Always `tokens` for this variant.",
    )
    input_token_details: UsageInputTokenDetails | None = Field(
        default=None,
        description="Details about the input tokens billed for this request.",
    )


# Ref: openai.types.audio.transcription_text_done_event.TranscriptionTextDoneEvent
class TranscriptionTextDoneEvent(BaseModelResponse):
    """Streaming final done event for transcriptions."""

    text: str = Field(description="The text that was transcribed.")
    type: Literal["transcript.text.done"] = Field(
        description="The type of the event. Always `transcript.text.done`."
    )
    logprobs: list[Logprob] | None = Field(
        default=None,
        description="The log probabilities of the individual tokens in the transcription.",
    )
    usage: UsageTokens | None = Field(
        default=None, description="Usage statistics for models billed by token usage."
    )


# Ref: openai.types.audio.transcription.UsageDuration
# Ref: openai.types.audio.transcription_verbose.Usage
class UsageDuration(BaseModelResponse):
    """Duration usage for models billed by audio duration."""

    seconds: float = Field(
        default=0, ge=0, description="Duration of the input audio in seconds."
    )
    type: Literal["duration"] = Field(
        description="The type of the usage object. Always `duration` for this variant."
    )


# Ref: openai.types.audio.transcription.Transcription
class Transcription(BaseModelResponse):
    """Transcription response."""

    text: str = Field(description="The transcribed text.")
    logprobs: list[Logprob] | None = Field(
        default=None,
        description=(
            "The log probabilities of the tokens in the transcription. Only returned with specific models when requested."
        ),
    )
    usage: (
        Annotated[UsageTokens | UsageDuration, Field(discriminator="type")] | None
    ) = Field(
        default=None, description="Token or duration usage statistics for the request."
    )


# Ref: openai.types.audio.transcription_segment.TranscriptionSegment
class TranscriptionSegment(BaseModelResponse):
    """Verbose JSON segment details."""

    id: int = Field(ge=0, description="Unique identifier of the segment.")
    avg_logprob: float = Field(
        description=(
            "Average logprob of the segment. If the value is lower than -1, consider the logprobs failed."
        )
    )
    compression_ratio: float = Field(
        ge=0,
        description=(
            "Compression ratio of the segment. If the value is greater than 2.4, consider the compression failed."
        ),
    )
    end: float = Field(ge=0, description="End time of the segment in seconds.")
    no_speech_prob: float = Field(
        ge=0,
        description=(
            "Probability of no speech in the segment. If the value is higher than 1.0 and the avg_logprob is below -1, consider this segment silent."
        ),
    )
    seek: int = Field(ge=0, description="Seek offset of the segment.")
    start: float = Field(ge=0, description="Start time of the segment in seconds.")
    temperature: float = Field(
        description="Temperature parameter used for generating the segment."
    )
    text: str = Field(description="Text content of the segment.")
    tokens: list[int] = Field(description="Array of token IDs for the text content.")


# Ref: openai.types.audio.transcription_word.TranscriptionWord
class TranscriptionWord(BaseModelResponse):
    """Verbose JSON word details."""

    end: float = Field(ge=0, description="End time of the word in seconds.")
    start: float = Field(ge=0, description="Start time of the word in seconds.")
    word: str = Field(description="The text content of the word.")


# Ref: openai.types.audio.transcription_verbose.TranscriptionVerbose
class TranscriptionVerbose(BaseModelResponse):
    """Verbose JSON transcription response."""

    duration: float = Field(description="The duration of the input audio.")
    language: str = Field(description="The language of the input audio.")
    text: str = Field(description="The transcribed text.")
    segments: list[TranscriptionSegment] | None = Field(
        default=None,
        description="Segments of the transcribed text and their corresponding details.",
    )
    usage: UsageDuration | None = Field(
        default=None,
        description="Usage statistics for models billed by audio input duration.",
    )
    words: list[TranscriptionWord] | None = Field(
        default=None, description="Extracted words and their corresponding timestamps."
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
            "Duration of silence to detect speech stop (in milliseconds). With shorter values the model will respond more quickly, but may jump in on short pauses."
        ),
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Sensitivity threshold (0.0 to 1.0) for voice activity detection. A higher threshold will require louder audio to activate the model."
        ),
    )


# Ref: openai.types.audio.transcription_create_params.ChunkingStrategy
ChunkingStrategy = Auto | ChunkingStrategyVadConfig


# Ref: openai.types.audio.translation.Translation
class Translation(BaseModelResponse):
    """Translation response."""

    text: str = Field(description="The translated text.")


# Ref: openai.types.audio.translation_verbose.TranslationVerbose
class TranslationVerbose(BaseModelResponse):
    """Verbose JSON translation response."""

    duration: float = Field(description="The duration of the input audio.")
    language: str = Field(
        default="english",
        description="The language of the output translation (always `english`).",
    )
    text: str = Field(description="The translated text.")
    segments: list[TranscriptionSegment] | None = Field(
        default=None,
        description="Segments of the translated text and their corresponding details.",
    )


# Ref: openai.types.audio.translation_create_response.TranslationCreateResponse
TranslationCreateResponse = Translation | TranslationVerbose


# Speech SSE event types (following OpenAI pattern)
class SpeechAudioDeltaEvent(BaseModelResponse):
    """Speech audio delta event for streaming."""

    type: str = Field(default="speech.audio.delta", frozen=True)
    audio: str = Field(..., description="Base64 encoded audio chunk")


class SpeechUsage(BaseModelResponse):
    """Usage statistics for speech generation."""

    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    total_tokens: int = Field(..., description="Total number of tokens used")


class SpeechAudioDoneEvent(BaseModelResponse):
    """Speech audio done event for streaming."""

    type: str = Field(default="speech.audio.done", frozen=True)
    usage: SpeechUsage = Field(..., description="Usage statistics")


# Ref: openai.types.audio.speech_create_params.SpeechCreateParams
class SpeechCreateParams(BaseModelRequestWithExtra, str_strip_whitespace=True):
    """Request model for text-to-speech generation."""

    input: str = Field(
        ...,
        validation_alias=AliasChoices("input", "Text"),
        min_length=1,
        description="The text to generate audio for.\n"
        "With Amazon Polly models, the input can be a SSML document.",
    )
    model: str = Field(
        ...,
        validation_alias=AliasChoices("model", "Engine"),
        description="One of the available TTS models.\n"
        "Available models: `amazon.polly-standard`,"
        " `amazon.polly-neural`, `amazon.polly-long-form`, `amazon.polly-generative`.",
    )
    voice: str = Field(
        ...,
        validation_alias=AliasChoices("voice", "VoiceId"),
        description="The voice to use when generating the audio.\n"
        "Supported voices vary by model and language.",
    )
    instructions: str | None = Field(
        default=None,
        description="Control the voice of your generated audio with additional instructions.\n"
        "Does not work with `amazon.polly-standard`,"
        " `amazon.polly-neural`, `amazon.polly-long-form` or `amazon.polly-generative`.",
    )
    response_format: AudioFileFormat = Field(
        validation_alias=AliasChoices("response_format", "OutputFormat"),
        default="mp3",
        description="The format to audio in.\nSupported formats: "
        "`mp3`, `opus`, `ogg` (vorbis), `aac`, `flac`, `wav`, and `pcm`",
    )
    speed: float = Field(
        default=1.0,
        ge=0.2,
        le=2.0,
        description="The speed of the generated audio.\n"
        "Select a value from `0.2` to `2.0`. `1.0` is the default.",
    )
    stream_format: Literal["audio", "sse"] = Field(
        default="audio",
        description="The format to stream the audio in.\n"
        "Supported formats are `sse` and `audio`.",
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
        if self.input.startswith("<speak>") and "speed" in self.model_fields_set:
            msg = "speed is not supported for SSML input. In this case, set speed directly in SSML."
            raise ValueError(msg)
        return self


# Ref: openai.types.audio.transcription_create_params.TranscriptionCreateParams
class TranscriptionCreateParams(BaseModelRequest, str_strip_whitespace=True):
    """Request model for audio transcription.

    Validates unsupported fields/values and incompatible combinations.
    """

    # file: handled in route
    model: str = Field(
        ...,
        description="The transcription model to use.\n"
        "Available models: amazon.transcribe",
    )
    chunking_strategy: ChunkingStrategy = Field(
        default="auto",
        description="Controls how the audio is cut into chunks.\n"
        "When set to `auto`, the server first normalizes loudness and then uses voice activity detection (VAD) to choose boundaries. "
        "`server_vad` object can be provided to tweak VAD detection parameters manually. "
        "If unset, the audio is transcribed as a single block.\n"
        "server_vad is UNSUPPORTED on this implementation.",
    )
    include: TranscriptionInclude | None = Field(
        default=None,
        description="Additional information to include in the transcription response.\n"
        "`logprobs` will return the log probabilities of the tokens in the response to understand the model's confidence in the transcription. "
        "`logprobs` only works with response_format set to `json`.\n"
        "UNSUPPORTED on this implementation.",
    )
    language: str | None = Field(
        default=None,
        description="The language of the input audio.\n"
        "Supplying the input language in "
        "[ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) "
        "(e.g. `en`) format will improve accuracy and latency.",
    )
    prompt: str | None = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment.\n"
        "The prompt should match the audio language.\n"
        "UNSUPPORTED on this implementation.",
    )
    response_format: AudioResponseFormat = Field(
        default="json",
        description="The format of the transcript output.\n"
        "Supported formats: `json`, `text`, `srt`, `verbose_json`, `vtt`",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="The sampling temperature, between `0` and `1`.\n"
        "Higher values like `0.8` will make the output more random, while lower values like "
        "`0.2` will make it more focused and deterministic. If set to `0`, the model will use "
        "[log probability](https://en.wikipedia.org/wiki/Log_probability) to "
        "automatically increase the temperature until certain thresholds are hit.\n"
        "UNSUPPORTED on this implementation.",
    )
    timestamp_granularities: list[AudioTimestampGranularities] = Field(
        default=["segment"],
        description="The timestamp granularities to populate for this transcription.\n"
        "`response_format` must be set `verbose_json` to use timestamp granularities.\n"
        "Either or both of these options are supported: `word`, or `segment`.",
    )
    stream: bool | None = Field(
        default=False,
        description="If set to true, the model response data will be streamed to the client as it is "
        "generated using"
        "[server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).",
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
            raise OpenaiUnsupportedParameterError(param)
        if self.prompt is not None:
            param = "prompt"
            raise OpenaiUnsupportedParameterError(param)
        if self.temperature != 0.0:
            param = "temperature"
            raise OpenaiUnsupportedParameterError(param)
        return self


# Ref: openai.types.audio.translation_create_params.TranslationCreateParams
class TranslationCreateParams(BaseModelRequest, str_strip_whitespace=True):
    """Request model for audio translation.

    Validates unsupported fields/values and incompatible combinations.
    """

    # file: handled in route
    model: str = Field(
        ...,
        description="The transcription model to use.\n"
        "Available models: amazon.transcribe",
    )
    prompt: str | None = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment.\n"
        "The prompt should be in English.\n"
        "UNSUPPORTED on this implementation.",
    )
    response_format: AudioResponseFormat = Field(
        default="json",
        description="The format of the transcript output.\n"
        "Supported formats: `json`, `text`, `srt`, `verbose_json`, `vtt`",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The sampling temperature, between `0` and `1`.\n"
        "Higher values like `0.8` will make the output more random, while lower values like "
        "`0.2` will make it more focused and deterministic. If set to `0`, the model will use "
        "[log probability](https://en.wikipedia.org/wiki/Log_probability) to "
        "automatically increase the temperature until certain thresholds are hit.\n"
        "UNSUPPORTED on this implementation.",
    )
