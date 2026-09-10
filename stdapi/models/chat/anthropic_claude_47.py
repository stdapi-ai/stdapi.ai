"""Anthropic Claude 4.7 chat model implementation."""

from typing import ClassVar

from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 4.7 chat model implementation."""

    __slots__ = ()

    MATCHER = "anthropic.claude-opus-4-7"
    SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED: ClassVar[bool] = (
        False  # "role 'system' is not supported on this model"
    )
