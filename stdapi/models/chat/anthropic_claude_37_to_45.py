"""Anthropic Claude 3.7 to 4.5 chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING

from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel

if TYPE_CHECKING:
    from stdapi.types import JsonMapping
    from stdapi.types.openai_chat_completions import ReasoningEffort

#: Reasoning models: Budget factor over the token max count
_REASONING_EFFORT_BUDGET_FACTOR: dict[ReasoningEffort, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 75.0,
    "xhigh": 1.0,
}

#: Minimal value for reasoning budget
_REASONING_BUDGET_MINIMAL = 1024

#: Maximal value for reasoning budget
_REASONING_BUDGET_MAXIMAL = 32768


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 3.7 to 4.5 chat model implementation."""

    MATCHER = re_compile(
        r"^anthropic\.claude-(?:opus-(?:4-5|4-1|4)|sonnet-(?:4-5|4)|haiku-4-5|3-7-sonnet)-2"
    )
    TOOL_BETA_FLAGS = MappingProxyType(
        {
            "computer": "computer-use-2025-01-24",
            "memory": "context-management-2025-06-27",
        }
    )

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        reasoning_effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Configure reasoning parameters for Claude 3.7-4.5.

        Uses budget-based reasoning configuration. If ``budget_tokens`` is not
        provided, it is calculated from ``reasoning_effort``.

        Args:
            additional_request_fields: Request fields to modify with reasoning config.
            reasoning_effort: The reasoning effort level.
            budget_tokens: Optional explicit token budget for reasoning.
            max_tokens: Maximum number of tokens allowed for the model.
        """
        if budget_tokens is None:
            budget_tokens = (
                _REASONING_BUDGET_MINIMAL
                if reasoning_effort == "minimal"
                else max(
                    _REASONING_BUDGET_MINIMAL,
                    int(
                        ((max_tokens or _REASONING_BUDGET_MAXIMAL) - 1)
                        * _REASONING_EFFORT_BUDGET_FACTOR[reasoning_effort or "high"]
                    ),
                )
            )
        additional_request_fields["reasoning_config"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
