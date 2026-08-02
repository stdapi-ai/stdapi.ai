"""Google Gemma models on Amazon Bedrock Mantle, Gemma 4 and later."""

from re import Pattern
from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """Google Gemma chat model (e.g. ``google.gemma-4-e2b``), Gemma 4 and later."""

    __slots__ = ()

    #: Matches Gemma 4 and future versions; Gemma 3 is a legacy open-weight model.
    MATCHER: ClassVar[Pattern[str]] = re_compile(r"^google\.gemma-(?!3)\d")

    #: Gemma 4 and later models support both OpenAI APIs.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset(
        {"chat_completions", "responses"}
    )

    #: Newer Mantle-only models answer on the /openai/v1 surface.
    SURFACE: ClassVar[Surface | None] = "/openai/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
