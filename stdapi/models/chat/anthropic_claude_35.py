"""Anthropic Claude 3.5 chat model implementation."""

from re import compile as re_compile

from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Anthropic Claude chat model implementation."""

    MATCHER = re_compile(r"^anthropic\.claude-3-5-(?:haiku|sonnet)-")
    PROMPT_CACHING_SUPPORTED = True
    PROMPT_CACHING_TOOL_SUPPORTED = True
