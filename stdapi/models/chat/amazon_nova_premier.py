"""Amazon Nova premier chat model implementation."""

from types import MappingProxyType

from stdapi.models.chat._amazon_nova import (
    NOVA_INLINE_MEDIA_LIMITS,
    NOVA_S3_LOCATION_MEDIA_TYPES,
)
from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Amazon Nova premier chat model implementation."""

    __slots__ = ()

    MATCHER = "amazon.nova-premier"
    PROMPT_CACHING_SUPPORTED = True
    S3_LOCATION_MEDIA_TYPES = NOVA_S3_LOCATION_MEDIA_TYPES
    INLINE_MEDIA_LIMITS = NOVA_INLINE_MEDIA_LIMITS
    SUPPORTED_SYSTEM_TOOLS = frozenset({"nova_grounding"})
    CANONICAL_TO_BEDROCK_TOOL_MAP = MappingProxyType({"web_search": "nova_grounding"})
