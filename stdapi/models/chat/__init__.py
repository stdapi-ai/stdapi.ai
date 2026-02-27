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
from typing import TYPE_CHECKING, Any

from stdapi.models import ModelBase, ModelDetails, get_model, load_model_plugins

if TYPE_CHECKING:
    from re import Pattern

    from sse_starlette import EventSourceResponse

    from stdapi.types.anthropic_messages import Message, MessageCreateParams
    from stdapi.types.openai_chat_completions import (
        ChatCompletion,
        CompletionCreateParams,
    )


class ChatModelBase[RequestT, ResponseT](ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific chat models."""

    @abstractmethod
    async def create_completion(
        self,
        model: ModelDetails,
        request: CompletionCreateParams,
        completion_id: str,
        created: int,
    ) -> ChatCompletion | EventSourceResponse:
        """Create a chat completion.

        Args:
            model: Model details for the chat model.
            request: Chat completion creation request following OpenAI spec.
            completion_id: Stable identifier for the completion.
            created: Unix timestamp (seconds) of the request.

        Returns:
            - ChatCompletion when stream is False.
            - AsyncGenerator streaming ChatCompletionChunk events when stream is True.
        """

    @abstractmethod
    async def create_message(
        self, model: ModelDetails, request: MessageCreateParams, message_id: str
    ) -> Message | EventSourceResponse:
        """Create a message using Anthropic Messages API format.

        Args:
            model: Model details for the chat model.
            request: Message creation request following Anthropic spec.
            message_id: Stable identifier for the message.

        Returns:
            - Message when stream is False.
            - EventSourceResponse streaming MessageStreamEvent events when stream is True.
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
