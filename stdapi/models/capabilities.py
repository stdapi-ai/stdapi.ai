"""Route capability registry for model-to-route matching.

Each route registers itself here at import time.
"""

from dataclasses import dataclass
from enum import IntFlag, auto


class Capability(IntFlag):
    """Bitfield of model operation capabilities."""

    __slots__ = ()

    TTS = auto()
    STT = auto()
    STT_TRANSLATE = auto()
    IMAGE_GENERATION = auto()
    IMAGE_EDITION = auto()
    IMAGE_VARIATION = auto()
    RERANK = auto()
    VIDEO_GENERATION = auto()
    COUNT_TOKENS = auto()
    REALTIME = auto()


@dataclass(frozen=True, slots=True)
class RouteCapability:
    """Descriptor for a single model-facing API route.

    Attributes:
        operation_id: FastAPI / MCP operation identifier.
        path: Full HTTP path of the route including any prefix.
        required_input_modality: Input modality a model must support.
        required_output_modality: Output modality a model must produce.
        required_capability: Optional capability flag the model class must declare.
        mcp_tool: Whether the route is exposed as an MCP tool.
    """

    operation_id: str
    path: str
    required_input_modality: str
    required_output_modality: str
    required_capability: Capability | None = None
    mcp_tool: bool = True


#: Live mapping of operation_id → RouteCapability, populated by route modules at import time.
_REGISTRY: dict[str, RouteCapability] = {}

#: Public alias — always reflects the current state of ``_REGISTRY``.
ROUTE_CAPABILITIES: dict[str, RouteCapability] = _REGISTRY


def register_route_capability(
    operation_id: str,
    path: str,
    required_input_modality: str,
    required_output_modality: str,
    required_capability: Capability | None = None,
    *,
    mcp_tool: bool = True,
) -> None:
    """Register a route's model requirements in the capability registry.

    Args:
        operation_id: FastAPI operation_id / MCP tool name.
        path: Full HTTP route path including any prefix.
        required_input_modality: Modality the model must accept as input.
        required_output_modality: Modality the model must produce as output.
        required_capability: Optional capability flag the model class must expose.
        mcp_tool: Whether the route is exposed as an MCP tool. False for a route
            with no OpenAPI operation, which no MCP tool is built from.
    """
    _REGISTRY[operation_id] = RouteCapability(
        operation_id=operation_id,
        path=path,
        required_input_modality=required_input_modality,
        required_output_modality=required_output_modality,
        required_capability=required_capability,
        mcp_tool=mcp_tool,
    )
