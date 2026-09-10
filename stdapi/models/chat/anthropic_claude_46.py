"""Anthropic Claude 4.6 chat model implementation."""

from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types.anthropic_messages import ThinkingEffort


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 4.6 chat model implementation."""

    __slots__ = ()

    MATCHER = re_compile(r"anthropic\.claude-(?:sonnet|opus)-4-6")
    REASONING_OVERRIDE: ClassVar[dict[Effort | None, ThinkingEffort]] = {
        "minimal": "low",
        "xhigh": "high",
    }
