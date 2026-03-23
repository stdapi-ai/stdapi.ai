"""Anthropic Claude 4.6+ chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)

if TYPE_CHECKING:
    from stdapi.models import ModelDetails
    from stdapi.types import JsonMapping
    from stdapi.types.openai_chat_completions import ReasoningEffort

# Anthropic reasoning effort values
AnthropicReasoning = Literal["low", "medium", "high", "max"]

#: OpenAI to Anthropic reasoning effort override
_REASONING_OVERRIDE: dict[ReasoningEffort | None, AnthropicReasoning] = {
    "minimal": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}

_REASONING_CONFIG: dict[str, str] = {"type": "adaptive"}


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 4.6+ chat model implementation."""

    MATCHER = "anthropic.claude-"
    ALIAS_MATCHER = re_compile(r"^anthropic\.(.+?)(?:-v\d+(?::\d+)?)?$")

    _DATE_SUFFIX = re_compile(r"^(.+)-(\d{8})$")
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

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:
        """Return API model name aliases mapped to model IDs.

        Extends the base aliases with:
        - For Claude >= 4, A date-stripped alias for the most recent dated variant
          (e.g. ``claude-haiku-4-5-20251001`` -> ``claude-haiku-4-5``).
        - For Claude < 4, a ``-latest`` suffixed alias for the most recent dated variant
          (e.g. ``claude-3-7-sonnet-20250219`` -> ``claude-3-7-sonnet-latest``).

        Args:
            all_models: All available models keyed by Bedrock model ID.

        Returns:
            A dict mapping model alias to model ID.
        """
        aliases = super().get_aliases(all_models)
        newest: dict[str, tuple[str, str]] = {}
        for alias, model_id in aliases.items():
            if match := cls._DATE_SUFFIX.match(alias):
                base, date = match[1], match[2]
                if date > newest.get(base, ("",))[0]:
                    newest[base] = (date, model_id)

        for base, (_, model_id) in newest.items():
            aliases.setdefault(
                f"{base}-latest" if base.startswith("claude-3-") else base, model_id
            )

        return aliases

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        reasoning_effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure reasoning parameters for Claude 4.6+.

        When ``budget_tokens`` is explicitly provided (> 0), uses budget-based
        reasoning. Otherwise uses adaptive reasoning with an optional effort level.

        Args:
            additional_request_fields: Additional request fields dict to update.
            reasoning_effort: The reasoning effort level.
            budget_tokens: Maximum token budget for reasoning.
            max_tokens: Unused.
        """
        if budget_tokens:
            additional_request_fields["reasoning_config"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
        else:
            additional_request_fields["reasoning_config"] = _REASONING_CONFIG  # type: ignore[assignment]
            if reasoning_effort:
                additional_request_fields["output_config"] = {
                    "effort": _REASONING_OVERRIDE.get(reasoning_effort, "high")
                }
