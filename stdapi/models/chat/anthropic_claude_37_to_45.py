"""Anthropic Claude 3.7 to 4.5 chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ThinkingDisplay

#: Reasoning models: Budget factor over the token max count
_REASONING_EFFORT_BUDGET_FACTOR: dict[Effort, float] = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "xhigh": 1.0,
    "max": 1.0,
}

#: Minimal value for reasoning budget
_REASONING_BUDGET_MINIMAL = 1024

#: Maximal value for reasoning budget
_REASONING_BUDGET_MAXIMAL = 32768


def reasoning_budget(
    reasoning_effort: Effort | None, max_tokens: int | None
) -> int | None:
    """Return the thinking budget an effort level maps to within an output limit.

    Claude 3.7 to 4.5 take a token budget rather than an effort level. The
    budget must be at least the minimal one and smaller than ``max_tokens``, so
    an output limit leaving no room for it yields no budget and a warning.

    Args:
        reasoning_effort: The reasoning effort level, ``high`` when unset.
        max_tokens: Maximum number of tokens allowed for the model.

    Returns:
        The budget in tokens, or ``None`` when reasoning cannot be enabled.
    """
    ceiling = _REASONING_BUDGET_MAXIMAL if max_tokens is None else max_tokens
    if ceiling <= _REASONING_BUDGET_MINIMAL:
        log_error_details(
            "The requested output limit leaves no room for the smallest "
            "reasoning budget this model takes: reasoning is not enabled "
            "for this request",
            level="warning",
        )
        return None
    if reasoning_effort == "minimal":
        return _REASONING_BUDGET_MINIMAL
    return max(
        _REASONING_BUDGET_MINIMAL,
        int(
            (ceiling - 1)
            * _REASONING_EFFORT_BUDGET_FACTOR.get(reasoning_effort or "high", 1.0)
        ),
    )


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 3.7 to 4.5 chat model implementation."""

    __slots__ = ()

    MATCHER = re_compile(
        r"^anthropic\.claude-(?:opus-(?:4-5|4-1|4)|sonnet-(?:4-5|4)|haiku-4-5|3-7-sonnet)-2"
    )
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
            "computer": "computer_20250124",
            "memory": "memory_20250818",
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
        display: ThinkingDisplay | None = None,
    ) -> None:
        """Configure reasoning parameters for Claude 3.7-4.5.

        Uses budget-based reasoning configuration. If ``budget_tokens`` is not
        provided, it is calculated from ``reasoning_effort``, and left unset when
        ``max_tokens`` leaves no room for the smallest budget Bedrock accepts.
        When ``enabled`` is ``False``, reasoning is explicitly disabled.

        Args:
            additional_request_fields: Request fields to modify with reasoning config.
            enabled: Whether reasoning is explicitly enabled.
            reasoning_effort: The reasoning effort level.
            budget_tokens: Optional explicit token budget for reasoning.
            max_tokens: Maximum number of tokens allowed for the model.
            display: Whether thinking text is summarized or omitted, or the
                model default when ``None``.
        """
        if not enabled:
            additional_request_fields["reasoning_config"] = {"type": "disabled"}
            return

        # A requested budget of 0 is a value, not an omission.
        if budget_tokens is None:
            budget_tokens = reasoning_budget(reasoning_effort, max_tokens)
            if budget_tokens is None:
                return
        reasoning_config: JsonMapping = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
        if display:
            reasoning_config["display"] = display
        additional_request_fields["reasoning_config"] = reasoning_config
