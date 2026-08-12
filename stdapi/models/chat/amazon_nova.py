"""Amazon Nova chat model implementation."""

from stdapi.models.chat._amazon_nova import (
    NOVA_INLINE_MEDIA_LIMITS,
    NOVA_S3_LOCATION_MEDIA_TYPES,
)
from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Amazon Nova-specific chat model implementation.

    Supports Nova-specific features:
    - Prompt caching for system and messages (but not tools)
    - Attachments read from storage instead of the request body
    """

    __slots__ = ()

    MATCHER = "amazon.nova-"
    PROMPT_CACHING_SUPPORTED = True
    S3_LOCATION_MEDIA_TYPES = NOVA_S3_LOCATION_MEDIA_TYPES
    INLINE_MEDIA_LIMITS = NOVA_INLINE_MEDIA_LIMITS
