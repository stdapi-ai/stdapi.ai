"""Anthropic Claude 3.5 chat model implementation."""

from re import compile as re_compile
from types import MappingProxyType

from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude 3.5 chat model implementation."""

    MATCHER = re_compile(r"^anthropic\.claude-3-5-(?:haiku|sonnet)-")
    TOOL_BETA_FLAGS = MappingProxyType({"computer": "computer-use-2024-10-22"})
