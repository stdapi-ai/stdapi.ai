"""Anthropic Claude 4.8+ chat model implementation."""

from re import compile as re_compile

from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel

#: Claude Opus/Sonnet/Haiku 4.8+ and 5+, which accept native mid-conversation system messages.
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

    @property
    def SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED(self) -> bool:  # type: ignore[override]  # noqa: N802
        """Whether Bedrock accepts native mid-conversation system messages."""
        return _SYSTEM_MESSAGE_AS_MESSAGES_MATCHER.match(self._model_id) is not None
