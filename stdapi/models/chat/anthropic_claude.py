"""Anthropic Claude 4.6+ chat model implementation."""

from typing import TYPE_CHECKING, Any, Literal

from stdapi.models.chat._default import ChatModel as _BaseChatModel

if TYPE_CHECKING:
    from stdapi.types.openai_chat_completions import ReasoningEffort

# Anthropic reasoning effort values
AnthropicReasoning = Literal["low", "medium", "high", "max"]

#: OpenAI to Deepseek override
_REASONING_OVERRIDE: dict[ReasoningEffort | None, AnthropicReasoning] = {
    "minimal": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}

_REASONING_CONFIG = {"type": "adaptive"}


class ChatModel(_BaseChatModel):
    """Anthropic Claude chat model implementation."""

    MATCHER = "anthropic.claude-"
    PROMPT_CACHING_SUPPORTED = True
    PROMPT_CACHING_TOOL_SUPPORTED = True

    def _req_configure_reasoning(
        self,
        *,
        reasoning_effort: ReasoningEffort | None,
        budget_tokens: int | None,
        max_tokens: int | None,  # noqa: ARG002
        additional_request_fields: dict[str, Any],
    ) -> None:
        """Configures the reasoning parameters for the system.

        Setting up constraints and adaptive reasoning levels based on the provided efforts and budget.

        Args:
            reasoning_effort: Specifies the level of reasoning effort
                to be applied. If None, defaults to "high".
            budget_tokens: Maximum number of tokens allowed in the budget. If None,
                no restrictions are applied.
            max_tokens: An optional argument, currently not used in the configuration.
            additional_request_fields: A dictionary to include additional request
                fields. This will be updated with the configured reasoning type and token budget.
        """
        self._validate_no_budget_tokens(budget_tokens)
        additional_request_fields["reasoning_config"] = _REASONING_CONFIG
        additional_request_fields["output_config"] = {
            "effort": _REASONING_OVERRIDE.get(reasoning_effort, "high")
        }
