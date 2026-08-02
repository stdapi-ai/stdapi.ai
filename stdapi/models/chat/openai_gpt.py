"""OpenAI GPT chat model implementation."""

from re import compile as re_compile

from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """OpenAI GPT-specific chat model implementation."""

    __slots__ = ()

    MATCHER = "openai.gpt-"
    ALIAS_MATCHER = re_compile(r"^openai\.(.+?)(?:-1:0)?$")
