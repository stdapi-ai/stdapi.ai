"""Open-weight model families on Amazon Bedrock Mantle (Chat Completions only).

Covers the long tail of open-weight providers whose models answer only the
Chat Completions API on the legacy ``/v1`` surface (Gemma 3, Qwen, GLM,
Mistral, DeepSeek, MiniMax, Kimi, Nemotron, Palmyra, ...).
"""

from re import Pattern
from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """Open-weight Mantle chat model (e.g. ``qwen.qwen3-32b``)."""

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[Pattern[str]] = re_compile(
        r"^(?:qwen|zai|mistral|deepseek|minimax|moonshotai|nvidia|writer)\."
        r"|^google\.gemma-3"
    )

    #: These models only support the Chat Completions API.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"chat_completions"})

    #: Legacy-catalog models answer on the /v1 surface.
    SURFACE: ClassVar[Surface | None] = "/v1"
