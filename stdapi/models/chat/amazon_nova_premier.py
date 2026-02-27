"""Amazon Nova premier chat model implementation."""

from types import MappingProxyType

from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Amazon Nova premier chat model implementation."""

    MATCHER = "amazon.nova-premier"
    PROMPT_CACHING_SUPPORTED = True
    SUPPORTED_SYSTEM_TOOLS = MappingProxyType({"web_search": "nova_grounding"})
