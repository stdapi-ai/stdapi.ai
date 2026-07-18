"""OpenAI GPT-5.x models on Amazon Bedrock Mantle (Responses API only)."""

from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """OpenAI GPT-5.x chat model (e.g. ``openai.gpt-5.6-sol``)."""

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str] = "openai.gpt-5"

    #: GPT-5.x models are served exclusively by the Responses API.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"responses"})

    #: Newer Mantle-only models answer on the /openai/v1 surface.
    SURFACE: ClassVar[Surface | None] = "/openai/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
