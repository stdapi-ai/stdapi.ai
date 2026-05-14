"""Anthropic Claude 3.5 chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2024,
    AnthropicClaudeChatModel,
)

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 3.5 chat model implementation."""

    MATCHER = re_compile(r"^anthropic\.claude-3-5-(?:haiku|sonnet)-")
    TOOL_BETA_FLAGS = MappingProxyType(
        {
            "bash": _BETA_COMPUTER_USE_2024,
            "str_replace_editor": _BETA_COMPUTER_USE_2024,
            "str_replace_based_edit_tool": _BETA_COMPUTER_USE_2024,
            "computer": _BETA_COMPUTER_USE_2024,
        }
    )
    SERVER_TOOL_NAME_TO_TYPE = MappingProxyType(
        {
            "bash": "bash_20250124",
            "str_replace_based_edit_tool": "text_editor_20250728",
            "str_replace_editor": "text_editor_20241022",
            "computer": "computer_20241022",
        }
    )

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        *,
        enabled: bool,
        reasoning_effort: Effort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """No reasoning support for Anthropic Claude 3.5.

        Args:
            additional_request_fields: Additional request fields dict to update.
            enabled: Whether reasoning is explicitly enabled.
            reasoning_effort: The reasoning effort level.
            budget_tokens: Maximum token budget for reasoning.
            max_tokens: Unused.
        """
