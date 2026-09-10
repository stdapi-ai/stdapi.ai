"""Realtime speech-to-speech models base classes and dynamic registry.

Modules of this package define a ``RealtimeModel`` class with a ``MATCHER``
(string prefix or compiled regex) and are auto-loaded once on import. A model
class owns one backend's live conversation protocol and reports it as the
neutral events below; nothing here knows the client dialect those events are
rendered into.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from stdapi.api_errors import ApiError
from stdapi.models import ModelBase, get_model, load_model_plugins
from stdapi.models.capabilities import Capability

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Buffer, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager
    from re import Pattern

    from types_aiobotocore_bedrock.literals import RegionName


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """The backend detected the start of the caller's speech."""

    offset_ms: int = 0


@dataclass(frozen=True, slots=True)
class SpeechStopped:
    """The backend detected the end of the caller's speech."""

    offset_ms: int = 0


@dataclass(frozen=True, slots=True)
class InputTranscript:
    """A transcript of what the caller said."""

    text: str


@dataclass(frozen=True, slots=True)
class OutputTranscript:
    """What the model answered, as text.

    The transcript of the speech it is generating, or the answer itself in a
    session that produces no speech.
    """

    text: str


@dataclass(frozen=True, slots=True)
class OutputAudio:
    """One chunk of the model's spoken answer, as backend-rate 16-bit mono PCM."""

    audio: bytes


@dataclass(frozen=True, slots=True)
class ResponseStarted:
    """The model began answering."""


@dataclass(frozen=True, slots=True)
class ResponseFinished:
    """The model finished answering.

    Attributes:
        interrupted: Whether the caller spoke over the answer instead of it
            running to its end.
    """

    interrupted: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asked for one of the session's tools to be run.

    Attributes:
        call_id: Identifier the result is returned under.
        name: Tool the model called.
        arguments: Arguments it called the tool with, as a JSON string.
    """

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class UsageReport:
    """Everything the session has billed so far, as running totals."""

    input_speech_tokens: int = 0
    input_text_tokens: int = 0
    output_speech_tokens: int = 0
    output_text_tokens: int = 0
    total_tokens: int = 0


#: Every event a realtime backend reports.
BackendEvent = (
    SpeechStarted
    | SpeechStopped
    | InputTranscript
    | OutputTranscript
    | OutputAudio
    | ResponseStarted
    | ResponseFinished
    | ToolCall
    | UsageReport
)


@dataclass(frozen=True, slots=True)
class RealtimeTool:
    """One tool the model may call during the conversation.

    Attributes:
        name: Name the model calls the tool by.
        description: What the tool does, which is how the model decides to
            call it.
        parameters: JSON Schema of the arguments it takes, if it takes any.
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NamedTool:
    """The one tool every answer must start by calling.

    Attributes:
        name: Name of that tool.
    """

    name: str


#: How the model picks among the session's tools.
ToolChoice = Literal["auto", "required"] | NamedTool


class RealtimeBackendSession:
    """One live conversation with a backend model.

    Attributes:
        region: Region serving the session, recorded in usage and in the log.
    """

    __slots__ = ()

    region: RegionName | None = None

    async def send_audio(self, audio: Buffer) -> None:
        """Send one chunk of the caller's speech.

        Args:
            audio: 16-bit mono PCM at the session's input sample rate.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    async def send_text(self, text: str) -> None:
        """Add a written message from the caller to the conversation.

        Args:
            text: What the caller wrote.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    async def end_turn(self) -> None:
        """End the caller's turn, which is what starts the model answering.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    async def send_tool_result(self, call_id: str, output: str) -> None:
        """Answer one tool the model called.

        Args:
            call_id: Identifier of the call being answered.
            output: What the tool returned.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    async def events(self) -> AsyncGenerator[BackendEvent]:
        """Yield what the backend reports until the session ends.

        Yields:
            One neutral event per backend event that carries meaning.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError
        yield  # pragma: no cover - unreachable, declares the generator


class RealtimeModelBase[RequestT, ResponseT](ModelBase[RequestT, ResponseT]):
    """Base class for models answering on a live bidirectional conversation."""

    __slots__ = ()

    #: The session protocol takes no guardrail: ApplyGuardrail checks each turn.
    NATIVE_GUARDRAIL_SUPPORTED: ClassVar[bool] = False

    #: Sample rates, in hertz, the backend accepts for the caller's speech.
    INPUT_SAMPLE_RATES: ClassVar[frozenset[int]] = frozenset()

    #: Sample rates, in hertz, the backend can speak at.
    OUTPUT_SAMPLE_RATES: ClassVar[frozenset[int]] = frozenset()

    #: Voice used when the request names none.
    DEFAULT_VOICE: ClassVar[str] = ""

    #: Longest a session may last, in seconds, under the backend's own limit.
    MAX_SESSION_SECONDS: ClassVar[float] = 0.0

    #: Whether the model can call the tools a session declares.
    TOOLS_SUPPORTED: ClassVar[bool] = False

    def open_session(
        self,
        *,
        instructions: str,  # noqa: ARG002
        input_sample_rate: int,  # noqa: ARG002
        output_sample_rate: int,  # noqa: ARG002
        voice: str | None = None,  # noqa: ARG002
        temperature: float | None = None,  # noqa: ARG002
        max_output_tokens: int | None = None,  # noqa: ARG002
        speech_output: bool = True,  # noqa: ARG002
        tools: Sequence[RealtimeTool] = (),  # noqa: ARG002
        tool_choice: ToolChoice = "auto",  # noqa: ARG002
    ) -> AbstractAsyncContextManager[RealtimeBackendSession]:
        """Open one live conversation.

        Args:
            instructions: System instructions opening the session.
            input_sample_rate: Rate, in hertz, of the audio the caller sends.
            output_sample_rate: Rate, in hertz, the model should speak at.
            voice: Voice the model answers with, or None for the default one.
            temperature: Optional sampling temperature.
            max_output_tokens: Optional cap on the tokens one answer may use.
            speech_output: Whether the model should speak its answers.
            tools: Tools the model may call, empty when it may call none.
            tool_choice: How the model picks among them.

        Returns:
            An async context manager yielding the open session.

        Raises:
            ApiError: Live conversations are not supported by this model.
        """
        msg = f"Realtime sessions are not supported by {self.model.id}"
        raise ApiError(msg)

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Report the realtime capability when the model class implements it.

        Returns:
            Capability flags for the operations this model implements.
        """
        if cls.open_session is not RealtimeModelBase.open_session:
            return Capability.REALTIME
        return Capability(0)


#: Realtime model registry: (matcher, class) pairs sorted by specificity.
_REALTIME_MODEL_REGISTRY: list[
    tuple[str | Pattern[str], type[RealtimeModelBase[Any, Any]]]
] = []

#: Realtime model instance cache.
_REALTIME_MODEL_CACHE: dict[str, RealtimeModelBase[Any, Any]] = {}


def get_realtime_model(model_id: str) -> RealtimeModelBase[Any, Any]:
    """Resolve the realtime model class matching the provided identifier.

    Args:
        model_id: The provider model identifier.

    Returns:
        The realtime model associated to the ``model_id``.

    Raises:
        LookupError: If no registered realtime model matches ``model_id``.
    """
    return get_model(
        model_id, _REALTIME_MODEL_CACHE, _REALTIME_MODEL_REGISTRY, __name__
    )


load_model_plugins(
    class_type=RealtimeModelBase,
    package_name=__name__,
    registry=_REALTIME_MODEL_REGISTRY,
)
