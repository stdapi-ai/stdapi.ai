"""Google Gemma 4 models on Amazon Bedrock Mantle."""

from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """Google Gemma 4 chat model (e.g. ``google.gemma-4-e2b``)."""

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str] = "google.gemma-4"

    #: Gemma 4 models support both OpenAI APIs.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset(
        {"chat_completions", "responses"}
    )

    #: Newer Mantle-only models answer on the /openai/v1 surface.
    SURFACE: ClassVar[Surface | None] = "/openai/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
