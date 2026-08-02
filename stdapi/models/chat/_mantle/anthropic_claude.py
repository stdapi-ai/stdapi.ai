"""Anthropic Claude models on Amazon Bedrock Mantle (Messages API only)."""

from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from re import Pattern

    from stdapi.aws_bedrock_mantle import MantleApi

#: Claude 4.8+/5+ families handling mid-conversation system messages natively.
_SYSTEM_MESSAGE_AS_MESSAGES_MATCHER = re_compile(
    r"^anthropic\.claude-(?:(?:fable|mythos)-"
    r"|(?:opus|sonnet|haiku)-(?:4-(?:[89]|\d{2})|[5-9]|\d{2})(?:\D|$))"
)


class ChatModel(MantleChatModel):
    """Anthropic Claude chat model (e.g. ``anthropic.claude-haiku-4-5``)."""

    __slots__ = ()

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str] = "anthropic."

    #: Claude models are served exclusively by the Anthropic Messages API.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"messages"})

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")

    #: Native on the 4.8+/5+ generations of every family, plus Fable and Mythos.
    SYSTEM_MESSAGE_AS_MESSAGES_MATCHER: ClassVar[Pattern[str] | None] = (
        _SYSTEM_MESSAGE_AS_MESSAGES_MATCHER
    )
