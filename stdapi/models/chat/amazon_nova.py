"""Amazon Nova chat model implementation."""

from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """Amazon Nova-specific chat model implementation.

    Supports Nova-specific features:
    - Prompt caching for system and messages (but not tools)
    """

    MATCHER = "amazon.nova-"
    PROMPT_CACHING_SUPPORTED = True
