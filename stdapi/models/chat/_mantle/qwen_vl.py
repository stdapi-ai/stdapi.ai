"""Qwen vision-language models on Amazon Bedrock Mantle."""

from re import Pattern
from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """Qwen vision-language chat model (e.g. ``qwen.qwen3-vl-235b-a22b-instruct``).

    Qwen names its vision-language line ``-vl``, versioned or not
    (``qwen-vl-max``); every other Qwen model answers an image part with a
    refusal, so the modality is declared for this line only. The exclusion in
    ``open_weight.ChatModel`` mirrors this pattern and changes with it.
    """

    __slots__ = ()

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[Pattern[str]] = re_compile(r"^qwen\.qwen[\d.]*-vl")

    #: These models only support the Chat Completions API.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"chat_completions"})

    #: Legacy-catalog models answer on the /v1 surface.
    SURFACE: ClassVar[Surface | None] = "/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
