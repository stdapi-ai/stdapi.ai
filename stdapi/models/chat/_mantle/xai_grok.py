"""xAI Grok models on Amazon Bedrock Mantle."""

from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """xAI Grok chat model (e.g. ``xai.grok-4.3``)."""

    __slots__ = ()

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str] = "xai."

    #: Grok models support both OpenAI APIs.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset(
        {"chat_completions", "responses"}
    )

    #: Newer Mantle-only models answer on the /openai/v1 surface.
    SURFACE: ClassVar[Surface | None] = "/openai/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
