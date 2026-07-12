"""Rerank models base classes and dynamic registry.

This package exposes the base interfaces for rerank models and provides a
minimal plugin/registry system that auto-loads model implementations located in
this package directory and resolves them by matching the Bedrock model
identifier.

Design:
- Model modules expose a class named `RerankModel` with a class variable
  `MATCHER` containing a string prefix or compiled regex matching model
  identifiers.
- The package auto-loads and registers these classes once on import.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from stdapi.models import ModelBase, get_model, load_model_plugins
from stdapi.models.capabilities import Capability

if TYPE_CHECKING:
    from re import Pattern

    from stdapi.types import JsonMapping


class RerankedDocument(BaseModel):
    """Relevance of a single input document.

    Attributes:
        index: Index of the document in the request's document list.
        relevance_score: Relevance score of the document for the query.
    """

    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Rerank response.

    Attributes:
        results: Documents ordered by decreasing relevance.
        search_units: Billed search units for the query.
    """

    results: list[RerankedDocument] = []
    search_units: int = 0


class RerankModelBase(ModelBase[Any, Any]):
    """Base class for provider-specific rerank models."""

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Return capability flags for route-based model matching.

        Returns:
            Capability flags.
        """
        return Capability.RERANK

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None,
        extra_params: JsonMapping,
    ) -> RerankResponse:
        """Rank documents by relevance to the query.

        Args:
            query: The search query.
            documents: Texts to compare to the query.
            top_n: Maximum number of results to return, or None for all.
            extra_params: Extra model parameters.

        Returns:
            Rerank response.
        """


#: Model ID (or pattern) to rerank model class registry.
_MODEL_REGISTRY: list[tuple[str | Pattern[str], type[RerankModelBase]]] = []
#: Instantiated rerank models by model ID.
_MODEL_CACHE: dict[str, RerankModelBase] = {}


def get_rerank_model(model_id: str) -> RerankModelBase:
    """Resolve the rerank model class matching the provided identifier.

    Args:
        model_id: The provider model identifier (e.g., "cohere.rerank-v3-5:0").

    Returns:
        The rerank model associated to the ``model_id``.

    Raises:
        UnsupportedModelError: If no registered rerank model matches ``model_id``.
    """
    return get_model(model_id, _MODEL_CACHE, _MODEL_REGISTRY, __name__)


load_model_plugins(
    class_type=RerankModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_MODEL_REGISTRY,
)
