"""Mistral 7b models implementation.

- mistral.mistral-7b-instruct-v0:2
- mistral.mixtral-8x7b-instruct-v0:1
"""

from re import compile as re_compile

from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Mistral 7b specific chat model implementation.

    These models don't support System prompt.
    """

    __slots__ = ()

    MATCHER = re_compile(r"^mistral\.mi[xs]tral-(?:8x)?7b-")
    SYSTEM_PROMPT_SUPPORTED = False
