"""Amazon Nova 2 chat model implementation."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from stdapi.models.chat._default import ChatModel as _BaseChatModel

if TYPE_CHECKING:
    from stdapi.types import JsonMapping
    from stdapi.types.openai_chat_completions import ReasoningEffort

# Nova reasoning effort values
NovaReasoning = Literal["low", "medium", "high"]

#: OpenAI to Deepseek override
_REASONING_OVERRIDE: dict[ReasoningEffort | None, NovaReasoning] = {
    "minimal": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


class ChatModel(_BaseChatModel):
    """Amazon Nova-specific chat model implementation."""

    MATCHER = "amazon.nova-2-"
    PROMPT_CACHING_SUPPORTED = True
    SUPPORTED_SYSTEM_TOOLS = frozenset({"nova_grounding"})
    ANTHROPIC_TOOL_NAME_MAP = MappingProxyType({"web_search": "nova_grounding"})

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        reasoning_effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure Nova reasoning parameters.

        Args:
            additional_request_fields: Mutated with the ``reasoningConfig`` entry.
            reasoning_effort: Effort level; defaults to ``"medium"`` when ``None``.
            budget_tokens: Not supported; raises if set.
            max_tokens: Unused.

        Raises:
            ApiError: If *budget_tokens* is not ``None``.
        """
        self._validate_no_budget_tokens(budget_tokens)
        additional_request_fields["reasoningConfig"] = {
            "type": "enabled",
            "maxReasoningEffort": _REASONING_OVERRIDE.get(reasoning_effort, "medium"),
        }
