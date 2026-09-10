"""Anthropic Claude Fable and Mythos chat model implementation."""

from re import compile as re_compile
from typing import ClassVar

from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude Fable and Mythos chat model implementation.

    Every version of these two model families, released or not, always reasons:
    adaptive thinking cannot be disabled, only its effort level configured, so an
    explicitly disabled reasoning configuration is rejected. Mythos (``mythos-5``,
    ``mythos-preview``) is served by the Bedrock Mantle endpoint only, and is
    matched here for the day it is also served by bedrock-runtime.
    """

    __slots__ = ()

    MATCHER = re_compile(r"^anthropic\.claude-(?:fable|mythos)-")
    REASONING_DISABLE_SUPPORTED: ClassVar[bool] = False
    SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED: ClassVar[bool] = True
