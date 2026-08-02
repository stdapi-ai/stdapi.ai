"""Anthropic Claude Fable and Mythos chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import ClassVar

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)


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
    TOOL_BETA_FLAGS = MappingProxyType(
        {
            "bash": _BETA_COMPUTER_USE_2025,
            "str_replace_editor": _BETA_COMPUTER_USE_2025,
            "str_replace_based_edit_tool": _BETA_COMPUTER_USE_2025,
            "computer": _BETA_COMPUTER_USE_2025,
            "memory": _BETA_CONTEXT_MANAGEMENT_2025,
        }
    )
    SERVER_TOOL_NAME_TO_TYPE = MappingProxyType(
        {
            "bash": "bash_20250124",
            "str_replace_based_edit_tool": "text_editor_20250728",
            "str_replace_editor": "text_editor_20250728",
            "computer": "computer_20251124",
            "memory": "memory_20250818",
        }
    )
