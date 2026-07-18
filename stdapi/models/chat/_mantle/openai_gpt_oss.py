"""OpenAI gpt-oss open-weight models on Amazon Bedrock Mantle."""

from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(MantleChatModel):
    """OpenAI gpt-oss chat model (e.g. ``openai.gpt-oss-120b``)."""

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str] = "openai.gpt-oss"

    #: gpt-oss models support both OpenAI APIs.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset(
        {"chat_completions", "responses"}
    )

    #: Legacy-catalog models answer on the /v1 surface.
    SURFACE: ClassVar[Surface | None] = "/v1"
