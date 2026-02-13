"""Anthropic Claude 3.7 to 4.5 chat model implementation."""

from re import compile as re_compile
from typing import TYPE_CHECKING, Any

from stdapi.models.chat._default import ChatModel as _BaseChatModel

if TYPE_CHECKING:
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


class ChatModel(_BaseChatModel):
    """Anthropic Claude-specific chat model implementation."""

    MATCHER = re_compile(
        r"^anthropic\.claude-(?:opus-(?:4-5|4-1|4)|sonnet-(?:4-5|4)|haiku-4-5|3-7-sonnet)-2"
    )
    PROMPT_CACHING_SUPPORTED = True
    PROMPT_CACHING_TOOL_SUPPORTED = True

    def _req_configure_reasoning(
        self,
        *,
        reasoning_effort: ReasoningEffort | None,
        budget_tokens: int | None,
        max_tokens: int | None,
        additional_request_fields: dict[str, Any],
    ) -> None:
        """Configure reasoning parameters for Anthropic Claude models.

        Claude uses budget-based reasoning configuration. If budget_tokens is not
        provided, it will be calculated from reasoning_effort.

        Args:
            reasoning_effort: The reasoning effort level (used if budget_tokens not provided).
            budget_tokens: Optional explicit token budget for reasoning.
            max_tokens: Maximum number of tokens allowed for the model.
            additional_request_fields: Request fields to modify with reasoning config.
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
