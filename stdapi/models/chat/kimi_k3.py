"""Moonshot Kimi K3 chat model implementation."""

from re import compile as re_compile

from stdapi.models.chat._reasoning_effort import ReasoningEffortChatModel


class ChatModel(ReasoningEffortChatModel):
    """Moonshot Kimi K3 and later chat model implementation.

    Probed on Converse: the model reasons by default, and only
    ``additionalModelRequestFields.reasoning.effort`` changes that, ``none``
    returning no reasoning at all. The Kimi K2 ``thinking`` toggle and flat
    ``reasoning_effort`` are accepted and silently ignored, so they are not sent.

    Prompt caching is automatic: every ``cachePoint`` is rejected, while a
    repeated prefix is read back from the cache and reported as cache read and
    write tokens, so no cache point is ever sent.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
    """

    __slots__ = ()

    #: Kimi K3 and later under either Bedrock provider prefix, disjoint from the Kimi K2 matcher.
    MATCHER = re_compile(r"^moonshot(?:ai)?\.kimi-k(?:[3-9]|[1-9]\d)")
