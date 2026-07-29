"""Embedding models base classes and dynamic registry.

This package exposes the base interfaces for embedding models and provides a
minimal plugin/registry system that auto-loads model implementations located in
this package directory and resolves them by matching the OpenAI/Bedrock model
identifier.

Design:
- Model modules expose a class named `EmbeddingModel` with a class variable
  `MATCHER` containing a string prefix or compiled regex matching model
  identifiers.
- The package auto-loads and registers these classes once on import.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from stdapi.models import ModelBase, RequestT, ResponseT, get_model, load_model_plugins

if TYPE_CHECKING:
    from re import Pattern

    from stdapi.input_file import InputFileUrl
    from stdapi.types import JsonMapping


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


class EmbeddingModelBase(ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific embedding models."""

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
