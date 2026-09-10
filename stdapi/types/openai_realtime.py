"""Local OpenAI-compatible Realtime types.

Only the session configuration and the client-secret envelope are modelled: the
session's own events are exchanged as JSON over the WebSocket, where a pydantic
model per event type would cost a validation pass on every audio frame.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue

from stdapi.types import BaseModelRequest, BaseModelResponse

#: Sample rate the ``audio/pcm`` format is defined at, in hertz.
PCM_SAMPLE_RATE = 24000

#: Sample rate both G.711 formats are defined at, in hertz.
G711_SAMPLE_RATE = 8000

#: Sample rate of each audio format, keyed by its media type.
FORMAT_SAMPLE_RATES: dict[str, int] = {
    "audio/pcm": PCM_SAMPLE_RATE,
    "audio/pcmu": G711_SAMPLE_RATE,
    "audio/pcma": G711_SAMPLE_RATE,
}

#: Longest time to live an ephemeral client secret may be minted with, in seconds.
MAX_CLIENT_SECRET_TTL = 7200

#: Time to live an ephemeral client secret gets when the request names none.
DEFAULT_CLIENT_SECRET_TTL = 600


# Ref: openai.types.realtime.realtime_audio_formats.AudioPCM
class AudioFormat(BaseModel):
    """Audio format of one direction of the session."""

    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = Field(
        default="audio/pcm",
        description=(
            "Audio encoding: 24 kHz 16-bit mono PCM, or G.711 at 8 kHz, "
            "little-endian and mono in every case."
        ),
    )
    rate: Literal[24000] | None = Field(
        default=None, description="Sample rate of 'audio/pcm', always 24000."
    )


# Ref: openai.types.realtime.realtime_audio_input_turn_detection
class TurnDetection(BaseModel):
    """Voice activity detection ending the caller's turn."""

    type: Literal["server_vad", "semantic_vad"] = Field(
        default="server_vad", description="Turn detection mode."
    )
    create_response: bool = Field(
        default=True,
        description=(
            "UNSUPPORTED: a detected end of speech always starts a response; "
            "send audio without turn detection to transcribe without answering."
        ),
    )
    interrupt_response: bool = Field(
        default=True,
        description=(
            "UNSUPPORTED: whether new speech interrupts a response in progress "
            "is decided by the model."
        ),
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="UNSUPPORTED: the detection sensitivity is not adjustable.",
    )
    prefix_padding_ms: int | None = Field(
        default=None,
        ge=0,
        description="UNSUPPORTED: audio kept before detected speech is fixed.",
    )
    silence_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="UNSUPPORTED: the silence ending a turn is not adjustable.",
    )
    idle_timeout_ms: int | None = Field(
        default=None,
        ge=0,
        description="UNSUPPORTED: no response is started on an idle caller.",
    )
    eagerness: Literal["low", "medium", "high", "auto"] | None = Field(
        default=None, description="UNSUPPORTED: semantic eagerness is not adjustable."
    )


# Ref: openai.types.realtime.audio_transcription.AudioTranscription
class InputAudioTranscription(BaseModel):
    """Transcription of what the caller said."""

    model: str | None = Field(
        default=None,
        description=(
            "UNSUPPORTED: the transcript comes from the session's own model. "
            "Set this object at all to receive transcription events."
        ),
    )
    language: str | None = Field(
        default=None,
        description=(
            "UNSUPPORTED: the spoken language is detected, not declared. State "
            "the expected language in the session's instructions instead."
        ),
    )
    prompt: str | None = Field(
        default=None,
        description=(
            "UNSUPPORTED: the transcription takes no vocabulary hint. State the "
            "expected terms in the session's instructions instead."
        ),
    )


class NoiseReduction(BaseModel):
    """Noise reduction applied to the incoming audio."""

    type: Literal["near_field", "far_field"] | None = Field(
        default=None, description="UNSUPPORTED: incoming audio is not filtered."
    )


class AudioInputConfig(BaseModel):
    """Configuration of the audio the caller sends."""

    format: AudioFormat = Field(
        default_factory=AudioFormat, description="Format of the incoming audio."
    )
    noise_reduction: NoiseReduction | None = Field(
        default=None, description="UNSUPPORTED: incoming audio is not filtered."
    )
    transcription: InputAudioTranscription | None = Field(
        default=None,
        description=(
            "Set to receive a transcript of the caller's speech as "
            "conversation.item.input_audio_transcription events."
        ),
    )
    turn_detection: TurnDetection | None = Field(
        default_factory=TurnDetection,
        description=(
            "Turn detection settings, or null to end every turn with "
            "input_audio_buffer.commit instead."
        ),
    )


class AudioOutputConfig(BaseModel):
    """Configuration of the audio the model sends back."""

    format: AudioFormat = Field(
        default_factory=AudioFormat, description="Format of the returned audio."
    )
    voice: str | None = Field(default=None, description="Voice the model answers with.")
    speed: float | None = Field(
        default=None,
        gt=0.0,
        description="UNSUPPORTED: the spoken response is not time-scaled.",
    )


class AudioConfig(BaseModel):
    """Configuration for input and output audio."""

    input: AudioInputConfig = Field(
        default_factory=AudioInputConfig, description="Incoming audio settings."
    )
    output: AudioOutputConfig = Field(
        default_factory=AudioOutputConfig, description="Returned audio settings."
    )


# Ref: openai.types.realtime.realtime_function_tool_param.RealtimeFunctionToolParam
class FunctionTool(BaseModel):
    """One function the model may call during the session."""

    type: Literal["function"] = Field(
        default="function", description="Kind of tool being declared."
    )
    name: str | None = Field(
        default=None,
        description=(
            "Name the model calls the function by; an entry without one is "
            "ignored, since nothing can be called."
        ),
    )
    description: str | None = Field(
        default=None,
        description="What the function does, which is how the model decides to call it.",
    )
    parameters: dict[str, JsonValue] | None = Field(
        default=None, description="JSON Schema of the arguments the function takes."
    )


# Ref: openai.types.realtime.realtime_tools_config_union.Mcp
class McpServerTool(BaseModel):
    """UNSUPPORTED: a remote MCP server the session would call tools on."""

    type: Literal["mcp"] = Field(description="Kind of tool being declared.")
    server_label: str | None = Field(
        default=None, description="Name the remote server is referred to by."
    )


#: Either kind of tool a session may declare; an entry with no ``type`` is a function.
SessionTool = FunctionTool | McpServerTool


# Ref: openai.types.responses.tool_choice_function.ToolChoiceFunction
class FunctionToolChoice(BaseModel):
    """The one function every answer must start by calling."""

    type: Literal["function"] = Field(
        default="function", description="Kind of tool being named."
    )
    name: str = Field(description="Name of that function.")


# Ref: openai.types.responses.tool_choice_mcp.ToolChoiceMcp
class McpToolChoice(BaseModel):
    """UNSUPPORTED: a remote MCP server's tool every answer would start with."""

    type: Literal["mcp"] = Field(description="Kind of tool being named.")
    server_label: str = Field(description="Name the remote server is referred to by.")
    name: str | None = Field(
        default=None, description="Name of that server's tool, if only one is meant."
    )


class RealtimeSessionConfig(BaseModel):
    """Configuration of a speech-to-speech Realtime session."""

    type: Literal["realtime"] = Field(
        default="realtime", description="Kind of session to open."
    )
    model: str | None = Field(default=None, description="Model serving the session.")
    audio: AudioConfig = Field(
        default_factory=AudioConfig, description="Input and output audio settings."
    )
    instructions: str | None = Field(
        default=None, description="System instructions guiding every answer."
    )
    output_modalities: list[Literal["text", "audio"]] | None = Field(
        default=None,
        description="Modalities the model answers with; 'text' suppresses speech.",
    )
    max_output_tokens: int | Literal["inf"] | None = Field(
        default=None, description="Cap on the tokens one answer may use."
    )
    include: list[str] | None = Field(
        default=None, description="UNSUPPORTED: no extra output fields are available."
    )
    tools: list[SessionTool] | None = Field(
        default=None,
        description=(
            "Functions the model may call, declared before the conversation "
            "opens and fixed for the rest of it; remote MCP servers are refused."
        ),
    )
    tool_choice: (
        Literal["auto", "none", "required"] | FunctionToolChoice | McpToolChoice | None
    ) = Field(
        default=None,
        description=(
            "How the model picks among the tools: 'none' declares none at all, "
            "'required' and a named function make every answer start with a call. "
            "Naming a remote MCP server's tool is accepted and ignored."
        ),
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description=(
            "UNSUPPORTED: how many tools one answer calls is the model's own decision."
        ),
    )
    prompt: JsonValue = Field(
        default=None, description="UNSUPPORTED: prompt templates are not available."
    )
    reasoning: JsonValue = Field(
        default=None, description="UNSUPPORTED: the session reports no reasoning."
    )
    tracing: JsonValue = Field(
        default=None, description="UNSUPPORTED: session traces are not published."
    )
    truncation: JsonValue = Field(
        default=None,
        description="UNSUPPORTED: the conversation is not truncated server-side.",
    )


class TranscriptionSessionConfig(BaseModel):
    """Configuration of a transcription-only Realtime session."""

    type: Literal["transcription"] = Field(
        default="transcription", description="Kind of session to open."
    )
    model: str | None = Field(default=None, description="Model serving the session.")
    audio: AudioConfig = Field(
        default_factory=AudioConfig, description="Input and output audio settings."
    )
    include: list[str] | None = Field(
        default=None, description="UNSUPPORTED: no extra output fields are available."
    )


#: Either session shape, told apart by its ``type``.
SessionConfig = Annotated[
    RealtimeSessionConfig | TranscriptionSessionConfig, Field(discriminator="type")
]


class ClientSecretExpiresAfter(BaseModel):
    """When a minted client secret stops being usable."""

    anchor: Literal["created_at"] = Field(
        default="created_at", description="Moment the lifetime is counted from."
    )
    seconds: int = Field(
        default=DEFAULT_CLIENT_SECRET_TTL,
        ge=10,
        le=MAX_CLIENT_SECRET_TTL,
        description="Seconds the secret stays usable for.",
    )


class ClientSecretCreateParams(BaseModelRequest):
    """Request body of the client secret creation route."""

    expires_after: ClientSecretExpiresAfter | None = Field(
        default=None, description="Lifetime of the minted secret."
    )
    session: SessionConfig | None = Field(
        default=None,
        description=(
            "Session configuration the secret carries, applied to every session "
            "opened with it; a session may still override it."
        ),
    )


class ClientSecretCreateResponse(BaseModelResponse):
    """A minted ephemeral client secret and the session it opens."""

    value: str = Field(description="The client secret to connect with.")
    expires_at: int = Field(
        description="Unix time, in seconds, after which the secret is refused."
    )
    session: SessionConfig = Field(description="Session configuration it carries.")
