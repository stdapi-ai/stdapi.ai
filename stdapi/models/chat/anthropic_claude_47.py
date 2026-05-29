"""Anthropic Claude 4.7 chat model implementation."""

from types import MappingProxyType

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 4.7 chat model implementation."""

    MATCHER = "anthropic.claude-opus-4-7"
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
