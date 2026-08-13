"""Embedding models base classes and dynamic registry.

Modules of this package define an ``EmbeddingModel`` class with a ``MATCHER``
(string prefix or compiled regex) matching the OpenAI/Bedrock model identifier,
and are auto-loaded once on import.
"""

from abc import abstractmethod
from asyncio import Semaphore, gather
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from stdapi.models import ModelBase, get_model, load_model_plugins

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable
    from re import Pattern

    from stdapi.input_file import InputFileUrl
    from stdapi.types import JsonMapping

#: Concurrent invocations per request, for models that embed one input per call.
_EMBED_CONCURRENCY = 8


class EmbeddingImageDescription(BaseModel):
    """Metadata describing an image input echoed back by some embedding providers.

    Attributes:
        format: Image format (e.g. "png", "jpeg").
        width: Image width in pixels.
        height: Image height in pixels.
        bit_depth: Image bit depth.
    """

    format: str
    width: int
    height: int
    bit_depth: int


class EmbeddingResponse(BaseModel):
    """Embedding response.

    Attributes:
        embeddings: List of embedding vectors (one per input).
        total_tokens: Total token count reported by the provider (if available).
        prompt_tokens: Prompt tokens count reported by the provider.
        images: Metadata of the embedded images, when echoed by the provider.
        embeddings_by_type: Embedding vectors keyed by quantization type (e.g.
            "float", "int8", "binary"), when the provider returns more than
            the default float vectors.
    """

    embeddings: list[list[float]] = []
    total_tokens: int = 0
    prompt_tokens: int = 0
    images: list[EmbeddingImageDescription] | None = None
    embeddings_by_type: dict[str, list[list[float | int]]] | None = None


class EmbeddingModelBase[RequestT, ResponseT](ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific embedding models."""

    __slots__ = ()

    #: InvokeModel rejects native guardrail kwargs; ApplyGuardrail covers the route.
    NATIVE_GUARDRAIL_SUPPORTED: ClassVar[bool] = False

    @property
    def max_input_characters(self) -> int:
        """Characters accepted in one text input, or 0 when none is documented."""
        return 0

    @abstractmethod
    async def embed_text(
        self,
        inputs: list[InputFileUrl | str],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> EmbeddingResponse:
        """Get embeddings for text.

        Args:
            inputs: Texts to embed.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            Embedding response.
        """

    @staticmethod
    async def _gather_bounded[T](invocations: Iterable[Awaitable[T]]) -> list[T]:
        """Await one invocation per input under the per-request concurrency bound.

        A model that embeds one input per call fans out a number of concurrent
        invocations the caller controls: a document split into chunks yields
        hundreds. The bound keeps such a request running at a sustainable rate
        instead of throttling itself against the backend.

        Args:
            invocations: Awaitables to run, one per input.

        Returns:
            Their results, in the order the awaitables were given.
        """
        semaphore = Semaphore(_EMBED_CONCURRENCY)

        async def _bounded(invocation: Awaitable[T]) -> T:
            """Await one invocation while holding a slot of the bound.

            Returns:
                Whatever the invocation returned.
            """
            async with semaphore:
                return await invocation

        return await gather(*(_bounded(invocation) for invocation in invocations))

_MODEL_REGISTRY: list[
    tuple[str | Pattern[str], type[EmbeddingModelBase[Any, Any]]]
] = []
_MODEL_CACHE: dict[str, EmbeddingModelBase[Any, Any]] = {}


def get_embedding_model(model_id: str) -> EmbeddingModelBase[Any, Any]:
    """Resolve the embedding model class matching the provided identifier.

    Args:
        model_id: The provider model identifier (e.g., "cohere.embed-english-v3").

    Returns:
        The embedding model associated to the ``model_id``.

    Raises:
        LookupError: If no registered embedding model matches ``model_id``.
    """
    return get_model(model_id, _MODEL_CACHE, _MODEL_REGISTRY, __name__)


load_model_plugins(
    class_type=EmbeddingModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_MODEL_REGISTRY,
)
