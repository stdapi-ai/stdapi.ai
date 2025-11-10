"""Custom Models API."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from stdapi.models import (
    ModelDetails,
    get_all_models_details_and_modalities,
    initialize_bedrock_models,
)
from stdapi.monitoring import log_request_params, log_response_params

router = APIRouter(prefix="", tags=["models"])


@router.get(
    "/available_models",
    summary="Lists all currently available models (extended)",
    description=(
        """Lists the currently available models with detailed metadata.\n\n
This endpoint is project-specific and returns extended ModelDetails including modalities and regions.\n\n
Filter examples:\n\n
- Filter by input modality (TEXT):\n
  curl -G $BASE/available_models --data-urlencode 'input_modalities=TEXT'\n\n
- Filter by output modality (IMAGE):\n
  curl -G $BASE/available_models --data-urlencode 'output_modalities=IMAGE'\n"""
    ),
    response_description="A list of extended ModelDetails objects",
    responses={
        200: {"description": "OK"},
        400: {"description": "Invalid modality filter."},
    },
)
async def list_models(
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
) -> list[ModelDetails]:
    """Lists the currently available models.

    Returns a detailed list of currently available models.

    Returns:
        Models list

    Raises:
        HTTPException: When unable to retrieve models from backend services (500)
    """
    log_request_params(
        {"input_modalities": input_modalities, "output_modalities": output_modalities}
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
    if modalities:
        modalities_models_ids: set[str] = set()
        for modality in modalities:
            modality = modality.strip().upper()
            try:
                modalities_models_ids |= models_by_modalities[modality]
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"No model matching {modality_type} modality: {modality}.",
                ) from None
        models_ids &= modalities_models_ids
