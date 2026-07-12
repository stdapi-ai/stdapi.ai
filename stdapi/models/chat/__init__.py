"""Chat models base classes and dynamic registry.

This package exposes the base interfaces for chat models and provides a
minimal plugin/registry system that auto-loads model implementations located in
this package directory and resolves them by matching the model identifier.

Design:
- Model modules expose a class named `ChatModel` with a class variable
  `MATCHER` containing a string prefix or compiled regex matching model
  identifiers.
- The package auto-loads and registers these classes once on import.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

from stdapi.models import ModelBase, get_model, load_model_plugins

if TYPE_CHECKING:
    from collections.abc import Callable
    from re import Pattern

    from sse_starlette import EventSourceResponse

    from stdapi.types.anthropic_messages import (
        Message,
        MessageCreateParams,
        ThinkingEffort,
    )
    from stdapi.types.openai import ResponseModeration
    from stdapi.types.openai_chat_completions import ChatCompletion, ReasoningEffort
    from stdapi.types.openai_chat_completions import (
        CompletionCreateParams as ChatCompletionCreateParams,
    )
    from stdapi.types.openai_completions import Completion, CompletionCreateParams
    from stdapi.types.openai_responses import Response, ResponseCreateParams

    #: Merged reasoning effort type
    Effort = ReasoningEffort | ThinkingEffort

    class ReasoningParams(TypedDict):
        """Reasoning parameters resolved from a route-specific request.

        Matches the keyword arguments of ``ModelBase._req_configure_reasoning``.
        """

        enabled: bool
        reasoning_effort: Effort | None
        budget_tokens: int | None
        max_tokens: int | None


class ChatModelBase[RequestT, ResponseT](ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific chat models."""

    @abstractmethod
    async def create_completion(
        self, request: ChatCompletionCreateParams, completion_id: str, created: int
    ) -> ChatCompletion | EventSourceResponse:
        """Create a chat completion.

        Args:
            request: Chat completion creation request following OpenAI spec.
            completion_id: Stable identifier for the completion.
            created: Unix timestamp (seconds) of the request.

        Returns:
            - ChatCompletion when stream is False.
            - AsyncGenerator streaming ChatCompletionChunk events when stream is True.
        """

    @abstractmethod
    async def create_text_completion(
        self, request: CompletionCreateParams, completion_id: str, created: int
    ) -> Completion | EventSourceResponse:
        """Handle a completion request (OpenAI ``POST /v1/completions``).

        Args:
            request: Completion creation request following the OpenAI spec.
            completion_id: Stable identifier for the completion.
            created: Unix timestamp (seconds) of the request.

        Returns:
            ``Completion`` when ``stream`` is ``False``, otherwise an
            ``EventSourceResponse`` streaming completion chunks.
        """

    @abstractmethod
    async def create_message(
        self, request: MessageCreateParams, message_id: str
    ) -> Message | EventSourceResponse:
        """Create a message using Anthropic Messages API format.

        Args:
            request: Message creation request following Anthropic spec.
            message_id: Stable identifier for the message.

        Returns:
            - Message when stream is False.
            - EventSourceResponse streaming MessageStreamEvent events when stream is True.
        """

    @abstractmethod
    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        moderation_builder: Callable[[], ResponseModeration | None] | None = None,
    ) -> Response | EventSourceResponse:
        """Create a response using the OpenAI Responses API format.

        Args:
            request: Responses API creation request.
            response_id: Stable identifier for the response.
            created_at: Unix timestamp of the request.
            moderation_builder: Optional callable building the response
                ``moderation`` field, invoked once the full guardrail trace
                is available (at stream end when streaming).

        Returns:
            - Response when stream is False.
            - EventSourceResponse streaming ResponseStreamEvent events when stream is True.
        """


# Chat Model Registry
_CHAT_MODEL_REGISTRY: list[
    tuple[str | Pattern[str], type[ChatModelBase[Any, Any]]]
] = []
_CHAT_MODEL_CACHE: dict[str, ChatModelBase[Any, Any]] = {}


def get_chat_model(model_id: str) -> ChatModelBase[Any, Any]:
    """Resolve the chat model class matching the provided identifier.

    Args:
        model_id: The provider model identifier (e.g., "amazon.nova-micro-v1:0").

    Returns:
        The chat model associated to the ``model_id``.

    Raises:
        LookupError: If no registered chat model matches ``model_id``.
    """
    return get_model(model_id, _CHAT_MODEL_CACHE, _CHAT_MODEL_REGISTRY, __name__)


load_model_plugins(
    class_type=ChatModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_CHAT_MODEL_REGISTRY,
)
