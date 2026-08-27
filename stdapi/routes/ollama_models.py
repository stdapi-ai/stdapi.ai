"""Ollama-compatible model discovery endpoints.

- GET  /api/tags — list the models this server serves
- POST /api/show — describe one model
- GET  /api/ps — list the models loaded in memory
- GET  /api/version — the Ollama API version this server is compatible with
"""

from asyncio import Lock
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends
from starlette.responses import Response

from stdapi.api_providers.ollama import OLLAMA_API_VERSION, TAG_OLLAMA
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import (
    catalog_generation,
    get_all_models_details,
    initialize_bedrock_models,
    validate_model,
)
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.ollama import (
    CHAT_CAPABILITIES,
    EMBEDDING_CAPABILITY,
    ListResponse,
    ModelDetailsSummary,
    ModelSummary,
    PsResponse,
    ShowRequest,
    ShowResponse,
    VersionResponse,
)

if TYPE_CHECKING:
    from stdapi.models import ModelDetails

router = APIRouter(
    prefix=f"{SETTINGS.ollama_routes_prefix}/api", tags=["Models", TAG_OLLAMA]
)

#: Path of the Ollama chat route, as models advertise it in ``supported_routes``.
_CHAT_PATH: str = f"{SETTINGS.ollama_routes_prefix}/api/chat"

#: Path of the Ollama embed route, as models advertise it in ``supported_routes``.
_EMBED_PATH: str = f"{SETTINGS.ollama_routes_prefix}/api/embed"

#: Reported when a model's release date is unknown, so clients still parse a date.
_UNKNOWN_DATE: str = datetime.fromtimestamp(0, UTC).isoformat()

#: Nothing is ever resident: models run on a serverless backend.
_NO_RUNNING_MODELS = PsResponse(models=[])

#: /api/tags route response cache
_LIST_RESPONSE = ListResponse(models=[])
#: Guards concurrent rebuilds of the cache above.
_LIST_LOCK = Lock()
#: Catalog generation the cache above was built from; -1 until it is.
_CATALOG_GENERATION = -1


def model_digest(model_id: str) -> str:
    """Return the stable identifier reported for a model.

    Ollama's digest addresses the content of a local model file. Nothing here
    has such a file, so the value is a synthetic but stable identifier derived
    from the model name: usable as a cache key, and never a claim about weights.

    Args:
        model_id: Canonical model identifier.

    Returns:
        A 64-character hexadecimal identifier.
    """
    return sha256(model_id.encode()).hexdigest()


def model_capabilities(model: ModelDetails) -> list[str]:
    """Derive the features a model advertises to an Ollama client.

    Best effort, as everywhere in this catalogue: the backend remains the
    authority, and a request is worth making even for a feature not listed here.
    ``thinking`` is never advertised — no per-model source for it exists, and a
    client would show a control that silently does nothing.

    Args:
        model: Catalogue entry for the model.

    Returns:
        The capability names, in Ollama's own spelling.
    """
    capabilities: list[str] = []
    if _CHAT_PATH in model.supported_routes:
        capabilities.extend(CHAT_CAPABILITIES)
    if _EMBED_PATH in model.supported_routes:
        capabilities.append(EMBEDDING_CAPABILITY)
    if not capabilities:
        # An input modality is what a served route accepts, never a route in
        # itself: advertising one alone would list a model no request can reach.
        return capabilities
    if "IMAGE" in model.input_modalities:
        capabilities.append("vision")
    if "SPEECH" in model.input_modalities:
        capabilities.append("audio")
    return capabilities


def _details(model: ModelDetails) -> ModelDetailsSummary:
    """Build the origin block of a model.

    Args:
        model: Catalogue entry for the model.

    Returns:
        The provider, and empty values everywhere a local model file would
        have supplied one.
    """
    return ModelDetailsSummary(family=model.provider, families=[model.provider])


def _modified_at(model: ModelDetails) -> str:
    """Return the date a model was published.

    Args:
        model: Catalogue entry for the model.

    Returns:
        The release date, or the epoch when it is unknown.
    """
    return (
        model.start_of_life_time.isoformat()
        if model.start_of_life_time
        else _UNKNOWN_DATE
    )


def format_model_summary(model: ModelDetails) -> ModelSummary:
    """Format a catalogue entry as an Ollama model list entry.

    Args:
        model: Catalogue entry for the model.

    Returns:
        The list entry, naming the model by its canonical identifier.
    """
    return ModelSummary(
        name=model.id,
        model=model.id,
        modified_at=_modified_at(model),
        size=0,
        digest=model_digest(model.id),
        details=_details(model),
    )


@router.get(
    "/tags",
    summary="List available models (Ollama format)",
    operation_id="ollama_tags",
    description=(
        "Lists the models this server can serve through the Ollama endpoints "
        "(Ollama Tags API).\n\n"
        "Models are named by their canonical identifier, which is the name to "
        "send as `model`; short names accepted elsewhere on this server keep "
        "working. `size` is always 0 and `digest` is a stable identifier "
        "derived from the model name, since no model file is stored here."
    ),
    response_description="The models available on this server.",
    responses={200: {"description": "The available models."}},
    response_model_exclude_none=True,
)
async def list_tags(_: Annotated[None, Depends(authenticate)] = None) -> ListResponse:
    """List the models the Ollama endpoints can serve.

    Returns:
        ListResponse holding one entry per available model.
    """
    global _LIST_RESPONSE, _CATALOG_GENERATION  # noqa: PLW0603
    await initialize_bedrock_models()
    # Compared rather than taken from the call above: a refresh that ran in the
    # background, or that another listing route triggered, changes the catalog
    # without this call ever seeing it.
    generation = catalog_generation()
    async with _LIST_LOCK:
        if generation != _CATALOG_GENERATION or not _LIST_RESPONSE.models:
            models = await get_all_models_details()
            _LIST_RESPONSE = ListResponse(
                models=[
                    format_model_summary(models[model_id])
                    for model_id in sorted(models)
                    if model_capabilities(models[model_id])
                ]
            )
            _CATALOG_GENERATION = generation
    return log_response_params(_LIST_RESPONSE, exclude={"models"})


# HEAD is undocumented upstream but registered by the Ollama server, and clients
# probe it for liveness. Kept out of the schema for that reason: it publishes no
# payload of its own, and duplicating the operation would duplicate its tool.
@router.head("/tags", include_in_schema=False)
async def head_tags(_: Annotated[None, Depends(authenticate)] = None) -> Response:
    """Answer a liveness probe on the model list.

    Returns:
        An empty 200 response.
    """
    return Response(status_code=200)


@router.post(
    "/show",
    summary="Show information about a model (Ollama format)",
    operation_id="ollama_show",
    description=(
        "Returns the details and capabilities of one model (Ollama Show API).\n\n"
        "`license`, `modelfile`, `template`, `parameters` and `system` describe "
        "a local model file and are omitted, as Ollama Cloud omits them for a "
        "cloud-hosted model; the parameter and quantization details are empty "
        "for the same reason. `capabilities` is a best-effort hint, not a "
        "contract."
    ),
    response_description="The model's details and capabilities.",
    responses={
        200: {"description": "The model details."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def show(
    request: ShowRequest, _: Annotated[None, Depends(authenticate)] = None
) -> ShowResponse:
    """Describe one model.

    Args:
        request: Show parameters following the Ollama API.

    Returns:
        ShowResponse holding the model's details and capabilities.

    Raises:
        ApiError: With 404 if the model does not exist.
    """
    log_request_params(request)
    model = await validate_model(request.requested_model(), bedrock_only=False)
    return log_response_params(
        ShowResponse(
            details=_details(model),
            capabilities=model_capabilities(model),
            modified_at=_modified_at(model),
        )
    )


@router.get(
    "/ps",
    summary="List the models loaded in memory (Ollama format)",
    operation_id="ollama_ps",
    description=(
        "Lists the models currently loaded in memory (Ollama Ps API).\n\n"
        "Always empty: models are served on demand and none is ever resident, "
        "so nothing has to be loaded before a request or unloaded after it."
    ),
    response_description="The models loaded in memory.",
    responses={200: {"description": "The loaded models."}},
    response_model_exclude_none=True,
)
async def ps(_: Annotated[None, Depends(authenticate)] = None) -> PsResponse:
    """List the models resident in memory.

    Returns:
        PsResponse with an empty list: nothing is ever resident.
    """
    return log_response_params(_NO_RUNNING_MODELS)


@router.get(
    "/version",
    summary="Get the Ollama API version (Ollama format)",
    operation_id="ollama_version",
    description=(
        "Returns the Ollama API version this server is compatible with (Ollama "
        "Version API).\n\n"
        "A compatibility declaration, not this server's own version: clients "
        "use it to decide which features of the Ollama API they may send."
    ),
    response_description="The Ollama API version.",
    responses={200: {"description": "The version."}},
    response_model_exclude_none=True,
)
async def version(_: Annotated[None, Depends(authenticate)] = None) -> VersionResponse:
    """Report the Ollama API version this server is compatible with.

    Returns:
        VersionResponse holding the version string.
    """
    return log_response_params(VersionResponse(version=OLLAMA_API_VERSION))


@router.head("/version", include_in_schema=False)
async def head_version(_: Annotated[None, Depends(authenticate)] = None) -> Response:
    """Answer a liveness probe on the version endpoint.

    Returns:
        An empty 200 response.
    """
    return Response(status_code=200)
