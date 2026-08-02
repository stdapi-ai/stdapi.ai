"""Amazon Nova premier chat model implementation."""

from types import MappingProxyType

from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Amazon Nova premier chat model implementation."""

    __slots__ = ()

    MATCHER = "amazon.nova-premier"
    PROMPT_CACHING_SUPPORTED = True
    SUPPORTED_SYSTEM_TOOLS = frozenset({"nova_grounding"})
    CANONICAL_TO_BEDROCK_TOOL_MAP = MappingProxyType({"web_search": "nova_grounding"})
