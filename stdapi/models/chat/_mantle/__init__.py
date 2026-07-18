"""Chat model families served by the Amazon Bedrock Mantle endpoint.

This package mirrors the classic chat plugin system with its own registry:
modules define a ``ChatModel`` class with a ``MATCHER`` and are auto-loaded.
``_default`` is the fallback for unknown Mantle models (Responses API binding).
"""

from typing import TYPE_CHECKING, Any

from stdapi.models import get_model, load_model_plugins
from stdapi.models.chat import ChatModelBase

if TYPE_CHECKING:
    from re import Pattern

#: Mantle chat model registry: (matcher, class) pairs sorted by specificity.
_MANTLE_CHAT_MODEL_REGISTRY: list[
    tuple[str | Pattern[str], type[ChatModelBase[Any, Any]]]
] = []

#: Mantle chat model instance cache.
_MANTLE_CHAT_MODEL_CACHE: dict[str, ChatModelBase[Any, Any]] = {}


def get_mantle_chat_model(model_id: str) -> ChatModelBase[Any, Any]:
    """Resolve the Mantle chat model instance matching *model_id*.

    Args:
        model_id: The Mantle model identifier (e.g. ``openai.gpt-5.6-luna``).

    Returns:
        The Mantle chat model associated to the ``model_id``.
    """
    return get_model(
        model_id, _MANTLE_CHAT_MODEL_CACHE, _MANTLE_CHAT_MODEL_REGISTRY, __name__
    )


load_model_plugins(
    class_type=ChatModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_MANTLE_CHAT_MODEL_REGISTRY,
)
