"""Chat models base classes and dynamic registry.

Model modules of this package expose a class named ``ChatModel`` with a class
variable ``MATCHER`` holding a string prefix or a compiled regex matching model
identifiers.  The package auto-loads and registers these classes once on import,
then resolves a model by matching its identifier.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

from stdapi.config import SETTINGS
from stdapi.models import (
    MANTLE_MODELS,
    ModelBase,
    get_model,
    is_mantle_served,
    is_marketplace_endpoint,
    load_model_plugins,
)
from stdapi.monitoring import REQUEST

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

    __slots__ = ()

    #: Replayed reasoning content must carry the signature the model issued with it.
    #: When True, a replayed reasoning block that has no signature is dropped.
    REASONING_SIGNATURE_REQUIRED: ClassVar[bool] = False

    def native_store_supported(self) -> bool:
        """Whether this model stores responses natively (Bedrock Mantle).

        Returns:
            False for Converse-backed models (the server's own storage is used).
        """
        return False

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


def serves_via_mantle(model_id: str) -> bool:
    """Whether this request should be served by the Bedrock Mantle endpoint.

    Args:
        model_id: The provider model identifier.

    Returns:
        True for Mantle-registered models, or when the (gated) per-request
        ``x-stdapi-service: bedrock-mantle`` header targets a dual-homed model.
    """
    if (
        SETTINGS.aws_bedrock_mantle_service_header
        and model_id in MANTLE_MODELS
        and (request := REQUEST.get(None)) is not None
        and request.headers.get("x-stdapi-service") == "bedrock-mantle"
    ):
        return True
    return is_mantle_served(model_id)


def get_chat_model(model_id: str) -> ChatModelBase[Any, Any]:
    """Resolve the chat model class matching the provided identifier.

    Mantle-served models (and dual-homed models targeted by the per-request
    service header) resolve from the Mantle family registry; every other
    model resolves from the classic Converse registry.

    A Marketplace model endpoint always resolves to the generic Converse
    implementation. A family class encodes what one *serverless* model does
    differently, while an endpoint's own divergences are its container's, which
    Amazon Bedrock has already mapped onto Converse -- so a family matched by
    accident on the listing name would apply the wrong divergences.

    Args:
        model_id: The provider model identifier (e.g., "amazon.nova-micro-v1:0").

    Returns:
        The chat model associated to the ``model_id``.

    Raises:
        LookupError: If no registered chat model matches ``model_id``.
    """
    if serves_via_mantle(model_id):
        # Imported here: the _mantle package subclasses ChatModelBase.
        from stdapi.models.chat._mantle import get_mantle_chat_model  # noqa: PLC0415

        return get_mantle_chat_model(model_id)
    registry = [] if is_marketplace_endpoint(model_id) else _CHAT_MODEL_REGISTRY
    return get_model(model_id, _CHAT_MODEL_CACHE, registry, __name__)


load_model_plugins(
    class_type=ChatModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_CHAT_MODEL_REGISTRY,
)
