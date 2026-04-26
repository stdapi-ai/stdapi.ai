"""Anthropic Claude 4.6 chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)

if TYPE_CHECKING:
    from stdapi.types.anthropic_messages import ThinkingEffort
    from stdapi.types.openai_chat_completions import ReasoningEffort


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 4.6 chat model implementation."""

    MATCHER = re_compile(r"anthropic\.claude-(?:sonnet|opus)-4-6")
    REASONING_OVERRIDE: ClassVar[dict[ReasoningEffort | None, ThinkingEffort]] = {
        "minimal": "low",
        "xhigh": "high",
    }
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
