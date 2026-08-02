"""Anthropic Claude Opus 5 and later chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import ClassVar

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude Opus 5 and later chat model implementation.

    Unlike the other Claude 4.6+ models, Opus 5 supports no computer use tool
    version, so ``computer`` is left as a regular custom tool instead of being
    promoted to a server tool Bedrock would reject. Later Opus major versions
    are assumed to keep that behavior until proven otherwise.
    """

    __slots__ = ()

    MATCHER = re_compile(r"^anthropic\.claude-opus-(?:[5-9]|\d\d)")
    SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED: ClassVar[bool] = True
    TOOL_BETA_FLAGS = MappingProxyType(
        {
            "bash": _BETA_COMPUTER_USE_2025,
            "str_replace_editor": _BETA_COMPUTER_USE_2025,
            "str_replace_based_edit_tool": _BETA_COMPUTER_USE_2025,
            "memory": _BETA_CONTEXT_MANAGEMENT_2025,
        }
    )
    SERVER_TOOL_NAME_TO_TYPE = MappingProxyType(
        {
            "bash": "bash_20250124",
            "str_replace_based_edit_tool": "text_editor_20250728",
            "str_replace_editor": "text_editor_20250728",
            "memory": "memory_20250818",
        }
    )
