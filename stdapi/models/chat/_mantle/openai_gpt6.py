"""OpenAI GPT-6 and later models on Amazon Bedrock Mantle."""

from re import Pattern
from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.chat._mantle._openai_gpt import OpenAIGptChatModel

if TYPE_CHECKING:
    from stdapi.aws_bedrock_mantle import MantleApi, Surface


class ChatModel(OpenAIGptChatModel):
    """OpenAI GPT-6 chat model (e.g. ``openai.gpt-6-luna``), and later versions.

    Unlike GPT-5, these answer both Chat Completions and Responses on the
    ``/openai/v1`` surface: GPT-6 Luna and Sol served both in ``us-east-1``
    (probed 2026-09-22), and the GPT-6 Astra card lists both. The models are
    dual-homed and served by Mantle by default wherever a configured Mantle
    Region lists them, which is far fewer Regions than bedrock-runtime.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html
    """

    __slots__ = ()

    #: Matches GPT-6 and every later version, with or without a ``daybreak-<edition>-`` qualifier.
    MATCHER: ClassVar[Pattern[str]] = re_compile(
        r"^openai\.gpt-(?:daybreak-\w+-)?(?:[6-9]|[1-9]\d)"
    )

    #: GPT-6 models answer both OpenAI APIs.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset(
        {"chat_completions", "responses"}
    )

    #: GPT-6 models answer on the /openai/v1 surface.
    SURFACE: ClassVar[Surface | None] = "/openai/v1"

    #: Vision-capable (image + text input).
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT", "IMAGE")
