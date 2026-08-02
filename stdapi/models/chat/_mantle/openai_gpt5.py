"""OpenAI numbered GPT models on Amazon Bedrock Mantle (Responses API only)."""

from re import Pattern
from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """OpenAI numbered GPT chat model (e.g. ``openai.gpt-5.6-sol``), GPT-5 and later."""

    __slots__ = ()

    #: Matches GPT-5 and future numbered versions, not the gpt-oss family.
    MATCHER: ClassVar[Pattern[str]] = re_compile(r"^openai\.gpt-\d")

    #: Numbered GPT models are served exclusively by the Responses API.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"responses"})

    #: Newer Mantle-only models answer on the /openai/v1 surface.
    SURFACE: ClassVar[Surface | None] = "/openai/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
