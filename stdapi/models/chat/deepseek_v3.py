"""DeepSeek V3 chat model implementation."""

from typing import TYPE_CHECKING, Literal

from stdapi.models.chat._default import ChatModel as _BaseChatModel

if TYPE_CHECKING:
    from stdapi.types import JsonMapping
    from stdapi.types.openai_chat_completions import ReasoningEffort

DeepseekReasoning = Literal["low", "medium", "high"]

#: OpenAI to Deepseek override
_REASONING_OVERRIDE: dict[ReasoningEffort | None, DeepseekReasoning] = {
    "minimal": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


class ChatModel(_BaseChatModel):
    """DeepSeek-specific chat model implementation."""

    MATCHER = "deepseek.v3"

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        reasoning_effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure reasoning parameters for DeepSeek models.

        DeepSeek uses string-based reasoning_config and only supports reasoning_effort.
        The budget_tokens parameter is not supported.

        Args:
            reasoning_effort: The reasoning effort level (required for DeepSeek).
            budget_tokens: Not supported for DeepSeek models.
            max_tokens: Not used for DeepSeek models.
            additional_request_fields: Request fields to modify with reasoning config.

        Raises:
            ApiError: If budget_tokens is provided.
        """
        self._validate_no_budget_tokens(budget_tokens)
        additional_request_fields["reasoning_config"] = _REASONING_OVERRIDE.get(
            reasoning_effort, "high"
        )
