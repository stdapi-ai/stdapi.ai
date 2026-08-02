"""Anthropic Claude 4.8+ chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType

from stdapi.models.chat._anthropic_claude import (
    _BETA_COMPUTER_USE_2025,
    _BETA_CONTEXT_MANAGEMENT_2025,
    AnthropicClaudeChatModel,
)

#: Claude 4.8+/5+ families accepting native mid-conversation system messages (live-verified
#: on 4.8/opus-5/sonnet-5; newer versions assumed to keep the capability).
_SYSTEM_MESSAGE_AS_MESSAGES_MATCHER = re_compile(
    r"^anthropic\.claude-(?:opus|sonnet|haiku)-(?:4-(?:[89]|\d{2})|[5-9]|\d{2})(?:\D|$)"
)


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 4.8+ chat model implementation.

    This catch-all also serves the legacy Claude 2 and 3 generations and the
    Sonnet/Haiku families, which reject mid-conversation system messages, hence
    the per-model-ID gate.
    """

    __slots__ = ()

    MATCHER = "anthropic.claude-"
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

    @property
    def SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED(self) -> bool:  # type: ignore[override]  # noqa: N802
        """Whether Bedrock accepts native mid-conversation system messages."""
        return _SYSTEM_MESSAGE_AS_MESSAGES_MATCHER.match(self._model_id) is not None
