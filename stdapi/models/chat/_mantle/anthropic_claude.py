"""Anthropic Claude models on Amazon Bedrock Mantle (Messages API only)."""

from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi


class ChatModel(MantleChatModel):
    """Anthropic Claude chat model (e.g. ``anthropic.claude-haiku-4-5``)."""

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str] = "anthropic."

    #: Claude models are served exclusively by the Anthropic Messages API.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"messages"})

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
