"""Custom Models API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from stdapi.api_errors import ApiError
from stdapi.auth import authenticate
from stdapi.models import (
    ModelDetails,
    get_all_models_details_and_modalities,
    initialize_bedrock_models,
)
from stdapi.monitoring import log_request_params, log_response_params

router = APIRouter(prefix="", tags=["Models"])


@router.get(
    "/search_models",
    summary="Search available models with optional filters",
    operation_id="search_models",
    description=(
        "Search the catalogue of currently available models and return extended metadata "
        "(modalities, supported API routes, MCP tool names, AWS regions, streaming support, "
        "legacy status). Supplements the standard `/v1/models` list.\n\n"
        "All filters are optional and combined with **AND** logic — only models matching every "
        "supplied filter are returned, sorted by ID.\n\n"
        "**Agent workflow:**\n"
        "1. Call this tool first to find the right model ID, then pass it to the target endpoint.\n"
        "2. Use `route` with either a route path **or** an MCP tool name — both are accepted "
        "transparently (e.g. `route=/v1/images/generations` and `route=openai_image_generation` "
        "return the same models).\n"
        "3. **Combine filters for multimodal tasks:** when a tool supports extended input modalities "
        "(e.g. images in `openai_chat_completion`), add `input_modalities=IMAGE` alongside "
        "`route` — this ensures the model supports *both* the route and the required modality. "
        "A model that only handles text would otherwise appear in a route-only search and then fail "
        "at request time.\n"
        "4. **Exclude legacy models:** Add `legacy=false` to skip deprecated models unless you "
        "specifically need one.\n\n"
        "**Examples:**\n"
        "- Text generation: `route=openai_chat_completion&legacy=false`\n"
        "- Vision (image input): `route=openai_chat_completion&input_modalities=IMAGE&legacy=false`\n"
        "- Audio understanding: `route=openai_chat_completion&input_modalities=SPEECH&legacy=false`\n"
        "- Embeddings: `route=openai_embedding&legacy=false`\n"
        "- Image generation: `route=openai_image_generation&legacy=false`\n\n"
        '**Note:** Audio *output* from `openai_chat_completion` (via `modalities=["text","audio"]`) '
        "is a model-specific capability not separately tracked — use a `route` search "
        "and verify audio output support in the model documentation."
    ),
    response_description="A list of extended model details sorted by model ID",
    response_model_exclude_none=True,
    responses={
        200: {"description": "OK"},
        400: {"description": "Invalid modality, route, or MCP tool filter."},
    },
)
async def search_models(
    input_modalities: Annotated[
        set[str] | None,
        Query(
            description="Filter by expected input modalities (e.g., TEXT, IMAGE, SPEECH)."
        ),
    ] = None,
    output_modalities: Annotated[
        set[str] | None,
        Query(
            description="Filter by expected output modalities (e.g., TEXT, IMAGE, AUDIO)."
        ),
    ] = None,
    route: Annotated[
        str | None,
        Query(
            description=(
                "Filter to models that support a specific route path "
                "(e.g. /v1/chat/completions) or MCP tool name (e.g. openai_chat_completion). "
                "Both formats are accepted transparently."
            )
        ),
    ] = None,
    region: Annotated[
        str | None,
        Query(
            description="Filter to models available in a specific AWS region (e.g. us-east-1)."
        ),
    ] = None,
    streaming: Annotated[
        bool | None,
        Query(
            description="Filter by streaming support (true = streaming only, false = non-streaming only)."
        ),
    ] = None,
    legacy: Annotated[
        bool | None,
        Query(
            description="Filter by legacy status (true = deprecated models only, false = non-deprecated models only)."
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> list[ModelDetails]:
    """Search the model catalogue with optional filters and return extended metadata.

    Args:
        input_modalities: Filter to models that accept these input modalities (e.g. TEXT, IMAGE).
        output_modalities: Filter to models that produce these output modalities (e.g. TEXT, IMAGE).
        route: Filter to models that support this route path (e.g. /v1/chat/completions) or MCP
            tool name (e.g. openai_chat_completion). Both formats are accepted transparently.
        region: Filter to models available in a specific AWS region.
        streaming: Filter by streaming support.
        legacy: Filter by legacy/deprecated status.

    Returns:
        Filtered and sorted list of model details.

    Raises:
        ApiError: When an unknown modality, route, or MCP tool filter is specified (400) or
            models cannot be retrieved from backend services (500).
    """
    log_request_params(
        {
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "route": route,
            "region": region,
            "streaming": streaming,
            "legacy": legacy,
        }
    )
    await initialize_bedrock_models()
    (
        models,
        models_output_modalities,
        models_input_modalities,
    ) = await get_all_models_details_and_modalities()
    models_ids = set(models.keys())
    _filter_by_modality(input_modalities, models_ids, models_input_modalities, "input")
    _filter_by_modality(
        output_modalities, models_ids, models_output_modalities, "output"
    )
    _filter_by_route_or_tool(route, models_ids, models)
    if region is not None:
        _filter_by_region(region, models_ids, models)
    if streaming is not None:
        models_ids &= {
            mid for mid, m in models.items() if m.response_streaming is streaming
        }
    if legacy is not None:
        models_ids &= {mid for mid, m in models.items() if (m.legacy is True) is legacy}
    return log_response_params([models[model_id] for model_id in sorted(models_ids)])


def _filter_by_modality(
    modalities: set[str] | None,
    models_ids: set[str],
    models_by_modalities: dict[str, set[str]],
    modality_type: str,
) -> None:
    """Filters the provided models based on specific modalities.

    Args:
        modalities:
            A set of modality names to filter the models by. If None, no filtering is applied.
        models_ids:
            A set of model identifiers to be filtered. This set is modified in place.
        models_by_modalities:
            A dictionary mapping modality names (as keys) to sets of corresponding model
            identifiers (as values).
        modality_type:
            A string representing the descriptive name or type of modality, used for error messages.
    """
    if not modalities:
        return
    matched: set[str] = set()
    for raw in modalities:
        modality = raw.strip().upper()
        if (ids := models_by_modalities.get(modality)) is None:
            msg = f"No model matching {modality_type} modality: {modality}."
            raise ApiError(msg)
        matched |= ids
    models_ids &= matched


def _filter_by_route_or_tool(
    value: str | None, models_ids: set[str], models: dict[str, ModelDetails]
) -> None:
    """Filter models to those supporting a specific route path or MCP tool name.

    Checks both ``supported_routes`` and ``supported_mcp_tools`` and accepts either
    format transparently (e.g. ``/v1/images/generations`` and ``openai_image_generation``
    both match the same set of models).

    Args:
        value: Route path or MCP tool name to filter by. No-op when ``None``.
        models_ids: Set of model identifiers to filter in place.
        models: All model details keyed by model ID.
    """
    if not value:
        return
    if not (
        matched := {
            mid
            for mid, m in models.items()
            if value in m.supported_routes or value in m.supported_mcp_tools
        }
    ):
        msg = f"No model supporting route or MCP tool: {value}."
        raise ApiError(msg)
    models_ids &= matched


def _filter_by_region(
    region: str, models_ids: set[str], models: dict[str, ModelDetails]
) -> None:
    """Filter models to those available in the specified AWS region.

    Args:
        region: AWS region name to filter by (e.g. ``us-east-1``).
        models_ids: Set of model identifiers to filter in place.
        models: All model details keyed by model ID.
    """
    if not (matched := {mid for mid, m in models.items() if region in m.regions}):
        msg = f"No model available in region: {region}."
        raise ApiError(msg)
    models_ids &= matched
